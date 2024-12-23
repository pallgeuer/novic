export NOVIC=/home/user/novic
export NENV=novic
export DATASETS="$NOVIC"/datasets


for DSET in CIFAR10; do
    for EMBEDDER in openclip:timm/ViT-B-16-SigLIP openclip:apple/DFN5B-CLIP-ViT-H-14-378; do
        ./train.py action=embedder_zero_shot cls_dataset_root="$DATASETS" cls_dataset="$DSET" embedder_spec="$EMBEDDER" batch_size_image=32
    done
done