sudo parted /dev/nvme1n1 --script mklabel gpt
sudo parted /dev/nvme1n1 --script mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/nvme1n1p1
sudo mount /dev/nvme1n1p1 /mnt/data
sudo chown $USER /mnt/data
sudo chmod 700 /mnt/data
mkdir /mnt/data/huggingface