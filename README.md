# Image Deblurring with GANs

## Project Overview
This project implements a Generative Adversarial Network (GAN) for image deblurring. The model is trained to restore sharp images from their blurred counterparts using a combination of adversarial and pixel-wise loss functions.

## Key Features
- **Generator Architecture**: Uses an encoder-decoder structure with residual blocks for high-quality image generation.
- **Discriminator**: Classifies between real (sharp) and generated images to guide the training process.
- **Loss Functions**: Combines adversarial loss (BCEWithLogits) and pixel-wise loss (L1) for balanced training.
- **Metrics**: Evaluates performance using PSNR and SSIM for quantitative assessment.

## Requirements
- Python 3.10+
- PyTorch
- torchvision
- numpy
- Pillow
- scikit-image
- matplotlib
- pandas
- seaborn
- xlsxwriter (for Excel report generation)

## Dataset
The model is trained on the GoPro Deblur dataset, which contains pairs of blurred and sharp images. The dataset is split into:
- Training set: `/kaggle/input/gopro-deblur/gopro_deblur/`
- Validation set: `/kaggle/input/validation/val/`
- Test set: `/kaggle/input/test-data/test/`

## Training
To train the model:
1. Initialize the generator and discriminator.
2. Configure the data loaders for training and validation sets.
3. Run the training loop with the specified number of epochs.
4. Monitor training and validation losses.

Example command:
```python
python train.py --epochs 80 --batch_size 6
```

## Evaluation
The model can be evaluated on the test set using:
```python
python evaluate.py --model_path /path/to/generator.pth --test_dir /path/to/test_set
```
This will compute and display the average PSNR and SSIM scores.

## Results
- Training and validation loss curves are saved as `loss_plot.png`.
- Sample outputs are saved in the working directory.
- Detailed metrics are exported to `deblur_results.xlsx` with charts.

## Sample Output
To generate a deblurred image:
```python
python predict.py --model_path /path/to/generator.pth --image_path /path/to/blurred_image.png
```

## Pretrained Models
A pretrained generator model is available at:
`/kaggle/input/finalmodel/pytorch/default/1/generatorepoch74.pth`
