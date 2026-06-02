# Image Inpainting

Three datasets are available. MNIST and CIFAR-10 come from torchvision and are quick to use.

FFHQ requires a download from HuggingFace, which will likely take a long time...

Three models were trained: Unet, Diffusion Model, and Flow Matching Model. Results are available in the '''/results/''' directory.

usage:

initialize python environment (i created conda env and manually installed pip)

'''

./scripts/run

'''

Alternatively,

'''

conda activate ... # optional?

pip install .

python src/cs166_final_project/main.py \
    --config="./config/mnist.yaml" \
    --data-dir="./data/" \
    --output-dir="./output/" \
    model.name="ddpm"

'''
