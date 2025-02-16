# Set path to novic directory
export NOVIC=/home/user/novic
echo $NOVIC

# Downloading text data
wget -P "$NOVIC/extras/data" https://www2.informatik.uni-hamburg.de/wtm/corpora/novic/captions_dataset.json

wget -P "$NOVIC/extras/world" https://www2.informatik.uni-hamburg.de/wtm/corpora/ovic_datasets/world_dataset.zip
unzip -q "$NOVIC/extras/world/world_dataset.zip" -d "$NOVIC/extras/world" && rm "$NOVIC/extras/world/world_dataset.zip"

wget -P "$NOVIC/extras/wiki" https://www2.informatik.uni-hamburg.de/wtm/corpora/ovic_datasets/wiki_dataset.zip
unzip -q "$NOVIC/extras/wiki/wiki_dataset.zip" -d "$NOVIC/extras/wiki" && rm "$NOVIC/extras/wiki/wiki_dataset.zip"

wget -P "$NOVIC/extras/val3k" https://www2.informatik.uni-hamburg.de/wtm/corpora/ovic_datasets/val3k_dataset.zip
unzip -q "$NOVIC/extras/val3k/val3k_dataset.zip" -d "$NOVIC/extras/val3k" && rm "$NOVIC/extras/val3k/val3k_dataset.zip"



# Downloading image data - CIFAR10

mkdir -p "$NOVIC"/datasets # create datasets directory
export DATASETS="$NOVIC"/datasets # set path to datasets
export CIFAR_ROOT="$DATASETS"/CIFAR # set path to CIFAR directory
wget -P "$CIFAR_ROOT" https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xf "$CIFAR_ROOT"/cifar-10-python.tar.gz -C "$CIFAR_ROOT" && rm "$CIFAR_ROOT"/cifar-10-python.tar.gz


