//! FLUX.2 Klein 4B generation, and reference-image inpainting with the
//! stable-diffusion.cpp edit models in [`EditModel`].
//!
//! https://github.com/huggingface/diffusers/blob/a37f6f8394ac2a7ee8360c3abea811efe54512b1/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py
//! https://github.com/huggingface/diffusers/blob/a37f6f8394ac2a7ee8360c3abea811efe54512b1/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py
//!
//! FLUX.3 Image Klein keeps the FLUX.2 VAE and swaps in a FLUX.3 transformer
//! and a Qwen3-VL 4B text encoder. The distilled checkpoint samples in 16
//! steps with CFG 1.0 under the native default scheduler:
//! https://github.com/leejet/stable-diffusion.cpp/blob/3be2378/docs/flux3.md
//!
//! Qwen-Image 2.1 Turbo uses its own VAE and a Qwen3-VL 8B text encoder whose
//! vision projector encodes the reference images. It samples Turbo's fixed
//! 8-step sigma schedule at CFG 1.0, with dimensions divisible by 32:
//! https://github.com/leejet/stable-diffusion.cpp/blob/3be2378/docs/qwen_image_2.1.md

mod model;
mod processor;

use anyhow::{Context, Result, ensure};
use fast_image_resize::{FilterType, ResizeAlg, ResizeOptions, Resizer};
use image::{DynamicImage, RgbImage};
use koharu_diffusion::{
    GuidanceParams, ImageGenerationParams, SampleMethod, SampleParams, Scheduler, VaeFormat,
};
use koharu_runtime::HuggingFaceFile;

use self::{
    model::{Model, ModelPaths},
    processor::Flux2ImageProcessor,
};

pub use self::processor::{Flux2KleinInpaintOptions, Flux2KleinOptions};

crate::model_repository!("unsloth/FLUX.2-klein-4B-GGUF" @ "0084d1df98e2e2137fe776d55170bc4792ec1d66" {
    TRANSFORMER_WEIGHTS = "flux-2-klein-4b-Q4_K_M.gguf"
});
crate::model_repository!("black-forest-labs/FLUX.2-small-decoder" @ "a3efc24f613ef42d9428af62fdbd6f5fd8856c4a" {
    VAE_WEIGHTS = "full_encoder_small_decoder.safetensors"
});
crate::model_repository!("unsloth/Qwen3-4B-GGUF" @ "22c9fc8a8c7700b76a1789366280a6a5a1ad1120" {
    TEXT_ENCODER_WEIGHTS = "Qwen3-4B-Q4_K_M.gguf"
});
crate::model_repository!("unsloth/flux-3-image-klein-GGUF" @ "12ef13ce920d53f45fe805093910c899668b9e76" {
    FLUX3_TRANSFORMER_WEIGHTS = "flux-3-image-klein-UD-Q4_K_XL.gguf"
});
crate::model_repository!("Qwen/Qwen3-VL-4B-Instruct-GGUF" @ "1cd86afb9a95c410a6038ab3b40d8b578c892266" {
    FLUX3_TEXT_ENCODER_WEIGHTS = "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
});

crate::model_repository!("unsloth/Qwen-Image-2.1-Turbo-GGUF" @ "5d1f9f759ea877ae556cda88dbec958b62ad3809" {
    QWEN_IMAGE_TRANSFORMER_WEIGHTS = "qwen-image-2.1-turbo-Q4_K_M.gguf"
});
crate::model_repository!("unsloth/Qwen-Image-2.1-FP8" @ "bb21a8ba7f0371c19ad1f72c82a5e8fc10f39939" {
    QWEN_IMAGE_VAE_WEIGHTS = "vae/qwen_image_2.1_vae_bf16.safetensors"
});
crate::model_repository!("unsloth/Qwen3-VL-8B-Instruct-GGUF" @ "b93a7ee713758252c555be4210c00540df954dc2" {
    QWEN_IMAGE_TEXT_ENCODER_WEIGHTS = "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
    QWEN_IMAGE_VISION_WEIGHTS = "mmproj-F16.gguf",
});

/// Qwen-Image 2.1 Turbo's distilled schedule, including the terminal sigma.
/// https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/main/model_index.json
const QWEN_IMAGE_TURBO_SIGMAS: [f32; 9] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568, 0.0,
];

/// Edit checkpoints that share the reference-image inpainting flow below.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditModel {
    Flux2Klein,
    Flux3Klein,
    QwenImageTurbo,
}

/// Assets and sampling settings for one [`EditModel`].
#[derive(Debug)]
struct Checkpoint {
    name: &'static str,
    transformer: HuggingFaceFile<'static>,
    text_encoder: HuggingFaceFile<'static>,
    vision: Option<HuggingFaceFile<'static>>,
    vae: HuggingFaceFile<'static>,
    vae_format: VaeFormat,
    scheduler: Scheduler,
    sigmas: &'static [f32],
    /// Width and height must be multiples of this.
    alignment: u32,
}

impl EditModel {
    fn checkpoint(self) -> Checkpoint {
        match self {
            Self::Flux2Klein => Checkpoint {
                name: "FLUX.2 Klein",
                transformer: TRANSFORMER_WEIGHTS,
                text_encoder: TEXT_ENCODER_WEIGHTS,
                vision: None,
                vae: VAE_WEIGHTS,
                vae_format: VaeFormat::Flux2,
                scheduler: Scheduler::Flux2,
                sigmas: &[],
                alignment: 16,
            },
            Self::Flux3Klein => Checkpoint {
                name: "FLUX.3 Klein",
                transformer: FLUX3_TRANSFORMER_WEIGHTS,
                text_encoder: FLUX3_TEXT_ENCODER_WEIGHTS,
                vision: None,
                vae: VAE_WEIGHTS,
                vae_format: VaeFormat::Flux2,
                scheduler: Scheduler::Auto,
                sigmas: &[],
                alignment: 16,
            },
            Self::QwenImageTurbo => Checkpoint {
                name: "Qwen-Image 2.1 Turbo",
                transformer: QWEN_IMAGE_TRANSFORMER_WEIGHTS,
                text_encoder: QWEN_IMAGE_TEXT_ENCODER_WEIGHTS,
                vision: Some(QWEN_IMAGE_VISION_WEIGHTS),
                vae: QWEN_IMAGE_VAE_WEIGHTS,
                vae_format: VaeFormat::Auto,
                scheduler: Scheduler::Auto,
                sigmas: &QWEN_IMAGE_TURBO_SIGMAS,
                alignment: 32,
            },
        }
    }

    /// Inpainting options tuned for this checkpoint. FLUX.3 and Qwen-Image
    /// regenerate the masked area fully: under their shifted schedules, partial
    /// strength leaves the source lettering visible.
    pub fn inpaint_options(self) -> Flux2KleinInpaintOptions {
        let defaults = Flux2KleinInpaintOptions::default();
        match self {
            Self::Flux2Klein => defaults,
            Self::Flux3Klein => Flux2KleinInpaintOptions {
                num_inference_steps: 16,
                strength: 1.0,
                ..defaults
            },
            // The sigma schedule fixes the step count.
            Self::QwenImageTurbo => Flux2KleinInpaintOptions {
                num_inference_steps: QWEN_IMAGE_TURBO_SIGMAS.len() - 1,
                strength: 1.0,
                ..defaults
            },
        }
    }
}

#[derive(Debug)]
pub struct Flux2Klein {
    model: Model,
}

impl Flux2Klein {
    pub async fn load(device: crate::Device) -> Result<Self> {
        let (transformer, text_encoder, vae) = tokio::try_join!(
            TRANSFORMER_WEIGHTS.resolve(),
            TEXT_ENCODER_WEIGHTS.resolve(),
            VAE_WEIGHTS.resolve(),
        )
        .context("failed to resolve FLUX.2 Klein model assets")?;
        let model = Model::new(
            &device,
            "FLUX.2 Klein",
            ModelPaths {
                transformer,
                text_encoder,
                vision: None,
                vae,
                vae_format: VaeFormat::Flux2,
            },
        )?;
        Ok(Self { model })
    }

    pub fn inference(
        &self,
        image: &[DynamicImage],
        prompt: &str,
        options: &Flux2KleinOptions,
    ) -> Result<Vec<RgbImage>> {
        ensure!(
            !prompt.contains('\0'),
            "prompt contains an interior NUL byte"
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );
        ensure!(
            options.num_images_per_prompt > 0,
            "num_images_per_prompt must be greater than zero"
        );

        let mut condition_images = Vec::with_capacity(image.len());
        for image in image {
            Flux2ImageProcessor::check_image_input(image)?;
            let mut image = image.clone();
            if u64::from(image.width()) * u64::from(image.height()) > 1024 * 1024 {
                image = Flux2ImageProcessor::_resize_to_target_area(&image, 1024 * 1024);
            }
            let width = (image.width() / 16) * 16;
            let height = (image.height() / 16) * 16;
            ensure!(width > 0 && height > 0);
            condition_images
                .push(Flux2ImageProcessor::_resize_and_crop(&image, width, height)?.to_rgb8());
        }

        let height = options
            .height
            .or_else(|| condition_images.first().map(RgbImage::height))
            .unwrap_or(1024);
        let width = options
            .width
            .or_else(|| condition_images.first().map(RgbImage::width))
            .unwrap_or(1024);
        let height = (height / 16) * 16;
        let width = (width / 16) * 16;
        ensure!(width > 0 && height > 0);

        self.model.forward(&ImageGenerationParams {
            prompt: prompt.to_owned(),
            width: i32::try_from(width)?,
            height: i32::try_from(height)?,
            reference_images: condition_images,
            reference_image_args: Some("resize_before_vae=0".into()),
            sample: SampleParams {
                guidance: GuidanceParams {
                    text_cfg: 1.0,
                    ..GuidanceParams::default()
                },
                scheduler: Scheduler::Flux2,
                sample_method: SampleMethod::Euler,
                sample_steps: options.num_inference_steps,
                ..SampleParams::default()
            },
            seed: options.seed,
            batch_count: options.num_images_per_prompt,
            ..ImageGenerationParams::default()
        })
    }
}

#[derive(Debug)]
pub struct Flux2KleinInpaint {
    model: Model,
    checkpoint: Checkpoint,
}

impl Flux2KleinInpaint {
    pub async fn load(device: crate::Device, edit_model: EditModel) -> Result<Self> {
        let checkpoint = edit_model.checkpoint();
        let name = checkpoint.name;
        let (transformer, text_encoder, vae, vision) = tokio::try_join!(
            checkpoint.transformer.resolve(),
            checkpoint.text_encoder.resolve(),
            checkpoint.vae.resolve(),
            async {
                match checkpoint.vision {
                    Some(vision) => vision.resolve().await.map(Some),
                    None => Ok(None),
                }
            },
        )
        .with_context(|| format!("failed to resolve {name} model assets"))?;
        let model = Model::new(
            &device,
            name,
            ModelPaths {
                transformer,
                text_encoder,
                vision,
                vae,
                vae_format: checkpoint.vae_format,
            },
        )?;
        Ok(Self { model, checkpoint })
    }

    pub fn inference(
        &self,
        prompt: &str,
        image: &DynamicImage,
        image_reference: Option<&DynamicImage>,
        mask_image: &DynamicImage,
        options: &Flux2KleinInpaintOptions,
    ) -> Result<DynamicImage> {
        ensure!(
            image.width() == mask_image.width() && image.height() == mask_image.height(),
            "image/mask dimensions differ: image={}x{}, mask={}x{}",
            image.width(),
            image.height(),
            mask_image.width(),
            mask_image.height()
        );
        ensure!(
            !prompt.contains('\0'),
            "prompt contains an interior NUL byte"
        );
        ensure!(
            options.strength > 0.0 && options.strength <= 1.0,
            "{} inpaint strength must be greater than zero and at most one",
            self.model.name
        );
        ensure!(
            options.num_inference_steps > 0,
            "num_inference_steps must be greater than zero"
        );

        let mut image = image.clone();
        if u64::from(image.width()) * u64::from(image.height()) > 1024 * 1024 {
            image = Flux2ImageProcessor::_resize_to_target_area(&image, 1024 * 1024);
        }
        let alignment = self.checkpoint.alignment;
        let width = (image.width() / alignment) * alignment;
        let height = (image.height() / alignment) * alignment;
        ensure!(width > 0 && height > 0);
        let source = image.to_rgb8();
        let mut image = RgbImage::new(width, height);
        Resizer::new()
            .resize(
                &source,
                &mut image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination images have the same pixel type");
        let source_mask = mask_image.to_luma8();
        let mut mask_image = image::GrayImage::new(width, height);
        Resizer::new()
            .resize(
                &source_mask,
                &mut mask_image,
                &ResizeOptions::new()
                    .resize_alg(ResizeAlg::Convolution(FilterType::Lanczos3))
                    .use_alpha(false),
            )
            .expect("source and destination masks have the same pixel type");

        let crop_coords = options.padding_mask_crop.and_then(|padding| {
            Flux2ImageProcessor::get_crop_region(&mask_image, width, height, padding)
        });
        let (init_image, mut native_mask) = if let Some((x1, y1, x2, y2)) = crop_coords {
            let image_crop = DynamicImage::ImageRgb8(
                image::imageops::crop_imm(&image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            let mask_crop = DynamicImage::ImageLuma8(
                image::imageops::crop_imm(&mask_image, x1, y1, x2 - x1, y2 - y1).to_image(),
            );
            (
                Flux2ImageProcessor::_resize_and_fill(&image_crop, width, height).to_rgb8(),
                Flux2ImageProcessor::_resize_and_fill(&mask_crop, width, height).to_luma8(),
            )
        } else {
            (image.clone(), mask_image.clone())
        };
        Flux2ImageProcessor::binarize(&mut native_mask);

        let mut reference_images = vec![init_image.clone()];
        if let Some(image_reference) = image_reference {
            let mut image_reference = image_reference.clone();
            if u64::from(image_reference.width()) * u64::from(image_reference.height())
                > 1024 * 1024
            {
                image_reference =
                    Flux2ImageProcessor::_resize_to_target_area(&image_reference, 1024 * 1024);
            }
            let reference_width = (image_reference.width() / alignment) * alignment;
            let reference_height = (image_reference.height() / alignment) * alignment;
            ensure!(reference_width > 0 && reference_height > 0);
            reference_images.push(
                Flux2ImageProcessor::_resize_and_crop(
                    &image_reference,
                    reference_width,
                    reference_height,
                )?
                .to_rgb8(),
            );
        }

        let strength = if options.strength >= 1.0 {
            1.0
        } else {
            let effective_steps = options.num_inference_steps
                - (options.num_inference_steps as f64 * (1.0 - options.strength)).floor() as usize;
            let boundary = effective_steps as f32 / options.num_inference_steps as f32;
            f32::from_bits(boundary.to_bits() - 1)
        };

        let generated = self
            .model
            .forward(&ImageGenerationParams {
                prompt: prompt.to_owned(),
                width: i32::try_from(width)?,
                height: i32::try_from(height)?,
                init_image: Some(init_image),
                reference_images,
                reference_image_args: Some("resize_before_vae=0".into()),
                mask_image: Some(native_mask),
                sample: SampleParams {
                    guidance: GuidanceParams {
                        text_cfg: 1.0,
                        ..GuidanceParams::default()
                    },
                    scheduler: self.checkpoint.scheduler,
                    sample_method: SampleMethod::Euler,
                    sample_steps: i32::try_from(options.num_inference_steps)?,
                    custom_sigmas: self.checkpoint.sigmas.to_vec(),
                    ..SampleParams::default()
                },
                seed: options.seed,
                batch_count: 1,
                strength,
                ..ImageGenerationParams::default()
            })?
            .into_iter()
            .next()
            .with_context(|| format!("{} returned no inpainted image", self.model.name))?;

        let generated =
            Flux2ImageProcessor::apply_overlay(&mask_image, &image, generated, crop_coords)?;
        Ok(DynamicImage::ImageRgb8(generated))
    }
}
