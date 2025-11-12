import os
import cv2
import numpy as np
import torch
from backbones import get_model
import random
import shutil

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class ImageProcessor:
    def __init__(self, base_dir, destination_dir, model_path, num_samples=300, min_images=5):
        self.base_dir = base_dir
        self.destination_dir = destination_dir
        self.num_samples = num_samples
        self.min_images = min_images

        self.net = self.load_model(model_path)
        self.sampled_ids = []

    def load_model(self, model_path):
        with torch.no_grad():
            net = get_model('r18', fp16=False)
            net.load_state_dict(torch.load(model_path))
            net.eval()
        return net

    def copy_image_to_folder(self, source_image_path, rank=None, is_target=False):
        filename = os.path.basename(source_image_path)
        if "_" in filename:
            folder_name = filename.split("_")[0]
            folder_path = os.path.join(self.destination_dir, folder_name)
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            # 파일 이름에 순위 또는 "target" 추가
            name, ext = os.path.splitext(filename)
            if is_target:
                new_filename = f"{name}_target{ext}"
            else:
                new_filename = f"{name}_{rank}{ext}"

            destination_path = os.path.join(folder_path, new_filename)
            shutil.copy(source_image_path, destination_path)
            print(f"파일 {filename} 이(가) {folder_path} 폴더로 '{new_filename}' 이름으로 복사되었습니다.")
        else:
            print(f"파일 {filename} 이름에 '_'가 없어 폴더를 생성하지 않았습니다.")

    def stratified_sampling(self, data, num_samples):
        n = len(data)
        if n < num_samples:
            num_samples = n
        indices = np.linspace(0, n - 1, num_samples, dtype=int)
        sampled_data = [data[i] for i in indices]
        return sampled_data

    def arcbook_imread(self, img_path):
        img = cv2.imread(img_path)
        height, width = img.shape[:2]

        top = int(0.01 * height)
        bottom = height - top
        left = int(0.01 * width)
        right = width - left
        img_cropped = img[top:bottom, left:right]

        img_resized = cv2.resize(img_cropped, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_transposed = np.transpose(img_rgb, (2, 0, 1))
        img_tensor = torch.from_numpy(img_transposed).unsqueeze(0).float()
        img_tensor.div_(255).sub_(0.5).div_(0.5)

        return img_tensor

    def get_highest_resolution_image(self, image_folder):
        max_resolution = 0
        target_image_path = None
        for img_name in os.listdir(image_folder):
            img_path = os.path.join(image_folder, img_name)
            img = cv2.imread(img_path)
            resolution = img.shape[0] * img.shape[1]
            if resolution > max_resolution:
                max_resolution = resolution
                target_image_path = img_path
        return target_image_path

    def calculate_distances(self, target_image, other_images):
        target_feat = self.net(target_image).detach().numpy()
        distances = []
        for img_path in other_images:
            img_feat = self.net(self.arcbook_imread(img_path)).detach().numpy()
            diff = np.subtract(target_feat, img_feat)
            dist = np.sum(np.square(diff), 1)
            distances.append((img_path, dist))
        return distances

    def process_id_folder(self, id_folder):
        images = [os.path.join(id_folder, img) for img in os.listdir(id_folder)]
        if len(images) < self.min_images:
            return None

        target_image_path = self.get_highest_resolution_image(id_folder)
        images.remove(target_image_path)

        target_image = self.arcbook_imread(target_image_path)
        distances = self.calculate_distances(target_image, images)

        distances.sort(key=lambda x: x[1], reverse=False)
        result = self.stratified_sampling(distances, 10)

        self.copy_image_to_folder(target_image_path, is_target=True)

        # 나머지 파일들 복사
        for rank, (img_path, _) in enumerate(result, 1):
            self.copy_image_to_folder(img_path, rank)

        return result

    def process(self):
        id_folders = os.listdir(self.base_dir)
        random.shuffle(id_folders)

        while len(self.sampled_ids) < self.num_samples:
            for id_folder in id_folders:
                id_folder_path = os.path.join(self.base_dir, id_folder)
                if os.path.isdir(id_folder_path):
                    distances = self.process_id_folder(id_folder_path)
                    if distances:
                        self.sampled_ids.append(id_folder)
                        if len(self.sampled_ids) >= self.num_samples:
                            break

        print(f"Total sampled ID folders: {len(self.sampled_ids)}")


if __name__ == "__main__":
    processor = ImageProcessor(
        base_dir="/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/target/",
        destination_dir="/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/sample_",
        model_path="./try9_model_best.pt",
        num_samples=4,
        min_images=5
    )
    processor.process()