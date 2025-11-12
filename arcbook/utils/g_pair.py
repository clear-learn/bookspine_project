#! encoding: utf-8

import os
import random

class GeneratePairs:
    """
    Generate the pairs.txt file that is used for training face classifier when calling python `src/train_softmax.py`.
    Or others' python scripts that needs the file of pairs.txt.

    Doc Reference: http://vis-www.cs.umass.edu/lfw/README.txt
    """

    def __init__(self, data_dir, pairs_filepath, img_ext, num_folds):
        """
        Parameter data_dir, is your data directory.
        Parameter pairs_filepath, where is the pairs.txt that belongs to.
        Parameter img_ext, is the image data extension for all of your image data.
        """
        self.data_dir = data_dir
        self.pairs_filepath = pairs_filepath
        self.img_ext = img_ext
        self.num_folds = num_folds


    def generate(self):
        for fold in range(self.num_folds):
            self._generate_matches_pairs(fold)
            self._generate_mismatches_pairs(fold)
        return


    def _generate_matches_pairs(self, fold):
        """
        Generate all matches pairs
        """
        for name in os.listdir(self.data_dir):
            if name == ".DS_Store":
                continue

            a = []
            for file in os.listdir(self.data_dir + name):
                if file == ".DS_Store":
                    continue
                a.append(file)

            with open(self.pairs_filepath, "a") as f:
                for i in range(5):
                    temp = random.choice(a).split("_") # This line may vary depending on how your images are named.
                    w = temp[0]
                    l = random.choice(a).split("_")[1].rstrip(self.img_ext)
                    r = random.choice(a).split("_")[1].rstrip(self.img_ext)
                    f.write(w + "\t" + l + "\t" + r + "\n")

        print('done')
        print(f"Done generating matches pairs for fold {fold}")


    def _generate_mismatches_pairs(self, fold):
        """
        Generate all mismatches pairs
        """
        for i, name in enumerate(os.listdir(self.data_dir)):
            if name == ".DS_Store":
                continue

            remaining = os.listdir(self.data_dir)
            remaining = [f_n for f_n in remaining if f_n != ".DS_Store"]
            # del remaining[i] # deletes the file from the list, so that it is not chosen again
            other_dir = random.choice(remaining)
            with open(self.pairs_filepath, "a") as f:
                for i in range(5):
                    file1 = random.choice(os.listdir(self.data_dir + name))
                    file2 = random.choice(os.listdir(self.data_dir + other_dir))
                    f.write(name + "\t" + file1.split("_")[1].rstrip(self.img_ext) + "\t" + other_dir + "\t" + file2.split("_")[1].rstrip(self.img_ext) + "\n")
        print('done')
        print(f"Done generating mismatches pairs for fold {fold}")
        return


if __name__ == '__main__':
    data_dir = "/media/yim/5ca43dd5-cbeb-4cae-aac0-e36cdd0808f7/book_side/cleansing_book/val/"
    pairs_filepath = "val_pairs.txt"
    img_ext = ".jpg"
    generatePairs = GeneratePairs(data_dir, pairs_filepath, img_ext, num_folds=10)
    generatePairs.generate()