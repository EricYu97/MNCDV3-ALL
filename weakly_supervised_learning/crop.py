import rasterio
import numpy as np
import os
import glob
from PIL import Image

def crop_and_save_patches(save_path, suffix_name, crop_size, step_size, image, mean=None, std=None):
    n_channels, height, width = image.shape
    h_patches, w_patches=(height-crop_size)//step_size, (width-crop_size)//step_size
    patch_num=0
    # print(height, width, crop_size, step_size, h_patches, w_patches)
    # print(image.shape, h_patches, w_patches)
    for h in range(h_patches+1):
        for w in range(w_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, w*step_size:w*step_size+crop_size]

            assert h*step_size+crop_size<=height and w*step_size+crop_size<=width
            
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if height%crop_size!=0:
        for w in range(w_patches+1):
            image_cropped=image[:, height-crop_size: height, w*step_size:w*step_size+crop_size]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if width%crop_size!=0:
        for h in range(h_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, width-crop_size: width]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            np.save(sample_save_path, image_cropped)
            patch_num=patch_num+1
    if height%crop_size!=0 and width%crop_size!=0:
        image_cropped=image[:, height-crop_size: height, width-crop_size: width]
        save_name=f'{suffix_name}_Patch{str(patch_num)}.npy'
        sample_save_path=os.path.join(save_path, save_name)
        if mean is not None and std is not None:
            image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
        np.save(sample_save_path, image_cropped)
        patch_num=patch_num+1
    return patch_num

def crop_and_save_patches_rgb_label(save_path, suffix_name, crop_size, step_size, image, mean=None, std=None):
    n_channels, height, width = image.shape
    h_patches, w_patches=(height-crop_size)//step_size, (width-crop_size)//step_size
    patch_num=0
    # print(height, width, crop_size, step_size, h_patches, w_patches)
    # print(image.shape, h_patches, w_patches)
    for h in range(h_patches+1):
        for w in range(w_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, w*step_size:w*step_size+crop_size]

            assert h*step_size+crop_size<=height and w*step_size+crop_size<=width
            
            save_name=f'{suffix_name}_Patch{str(patch_num)}.png'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            image_cropped=Image.fromarray(np.transpose(image_cropped, (1, 2, 0)))
            image_cropped.save(sample_save_path)
            patch_num=patch_num+1
    if height%crop_size!=0:
        for w in range(w_patches+1):
            image_cropped=image[:, height-crop_size: height, w*step_size:w*step_size+crop_size]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.png'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            image_cropped=Image.fromarray(np.transpose(image_cropped, (1, 2, 0)))
            image_cropped.save(sample_save_path)
            patch_num=patch_num+1
    if width%crop_size!=0:
        for h in range(h_patches+1):
            image_cropped=image[:, h*step_size:h*step_size+crop_size, width-crop_size: width]
            save_name=f'{suffix_name}_Patch{str(patch_num)}.png'
            sample_save_path=os.path.join(save_path, save_name)
            if mean is not None and std is not None:
                image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
            image_cropped=Image.fromarray(np.transpose(image_cropped, (1, 2, 0)))
            image_cropped.save(sample_save_path)
            patch_num=patch_num+1
    if height%crop_size!=0 and width%crop_size!=0:
        image_cropped=image[:, height-crop_size: height, width-crop_size: width]
        save_name=f'{suffix_name}_Patch{str(patch_num)}.png'
        sample_save_path=os.path.join(save_path, save_name)
        if mean is not None and std is not None:
            image_cropped=(image_cropped-mean[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]
        image_cropped=Image.fromarray(np.transpose(image_cropped, (1, 2, 0)))
        image_cropped.save(sample_save_path)
        patch_num=patch_num+1
    return patch_num


def get_domain_image_stats(root_path):
    # only directories are considered domains
    domains = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    domain_dict = {}
    missing_pre = 0
    missing_post = 0
    for domain in domains:
        pre_matches = sorted(glob.glob(os.path.join(root_path, domain, 'pre', '*.tif')))
        post_matches = sorted(glob.glob(os.path.join(root_path, domain, 'post', '*.tif')))

        pre_img_path = pre_matches[0] if pre_matches else None
        post_img_path = post_matches[0] if post_matches else None
        pre_label_path = os.path.join(root_path, domain, 'pre', 'label.png')
        post_label_path = os.path.join(root_path, domain, 'post', 'label.png')

        if pre_img_path is None:
            missing_pre += 1
        if post_img_path is None:
            missing_post += 1

        domain_dict[domain] = {
            'pre_img_path': pre_img_path,
            'post_img_path': post_img_path,
            'pre_label_path': pre_label_path if os.path.exists(pre_label_path) else None,
            'post_label_path': post_label_path if os.path.exists(post_label_path) else None,
        }

    print(f"Built domain dict for {len(domain_dict)} domains (missing_pre={missing_pre}, missing_post={missing_post})")
    # show a small sample for quick inspection
    for d, paths in list(domain_dict.items())[:5]:
        print(d, paths)

    return domain_dict

def read_image_per_domain(domain_info):
    pre_img_path = domain_info['pre_img_path']
    post_img_path = domain_info['post_img_path']
    pre_label_path = domain_info['pre_label_path']
    post_label_path = domain_info['post_label_path']

    pre_image = None
    post_image = None
    pre_label = None
    post_label = None

    if pre_img_path:
        with rasterio.open(pre_img_path) as src:
            pre_image = src.read()

    if post_img_path:
        with rasterio.open(post_img_path) as src:
            post_image = src.read()

    if pre_label_path:
        pre_label = np.array(Image.open(pre_label_path))
        pre_label=pre_label.transpose(2,0,1)

    if post_label_path:
        post_label = np.array(Image.open(post_label_path))
        post_label=post_label.transpose(2,0,1)

    return pre_image, post_image, pre_label, post_label

def compute_mean_std(domain_dict, max_images=None, scale_factor=1.0):
    """Compute per-channel mean and std across all pre and post images listed in domain_dict.

    Args:
        domain_dict: mapping domain -> {'pre_img_path', 'post_img_path', ...}
        max_images: optional int, stop after processing this many images (pre+post combined)

    Returns:
        mean: np.array with shape (C,) per-channel mean
        std:  np.array with shape (C,) per-channel std
    """
    total_pixels = 0
    sum_c = None
    sumsq_c = None
    min_c = None
    max_c = None
    channels = None
    images_processed = 0

    for domain, info in domain_dict.items():
        pre_img, post_img, _, _ = read_image_per_domain(info)
        for img in (pre_img, post_img):
            if img is None:
                continue
            # expect shape (C, H, W)
            if img.ndim != 3:
                print(f"Skipping image for domain {domain}: unexpected ndim={img.ndim}")
                continue
            C, H, W = img.shape
            if channels is None:
                channels = C
            elif C != channels:
                print(f"Skipping image for domain {domain}: channel mismatch (expected {channels}, got {C})")
                continue

            img = img.astype(np.float64) * scale_factor
            n_pix = H * W
            flat = img.reshape(C, -1)
            flat_sum = flat.sum(axis=1)
            flat_sumsq = (flat ** 2).sum(axis=1)
            flat_min = flat.min(axis=1)
            flat_max = flat.max(axis=1)

            if sum_c is None:
                sum_c = flat_sum
                sumsq_c = flat_sumsq
                min_c = flat_min
                max_c = flat_max
            else:
                sum_c += flat_sum
                sumsq_c += flat_sumsq
                min_c = np.minimum(min_c, flat_min)
                max_c = np.maximum(max_c, flat_max)

            total_pixels += n_pix
            images_processed += 1
            if max_images is not None and images_processed >= max_images:
                break
        if max_images is not None and images_processed >= max_images:
            break

    if total_pixels == 0:
        print("No image pixels processed, cannot compute mean/std/min/max")
        return None, None, None, None

    mean = sum_c / total_pixels
    var = (sumsq_c / total_pixels) - (mean ** 2)
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    return mean, std, min_c, max_c

def compute_mean_std_cropped(cropped_root, max_patches=None, scale_factor=1.0):
    """
    Compute per-channel mean and std across all .npy patches in cropped_root.
    Args:
        cropped_root: root directory containing cropped patches (e.g., './Cropped_MineNetCDV3_Processed_NEW')
        max_patches: optional int, stop after this many patches
    Returns:
        mean: np.array with shape (C,) per-channel mean
        std:  np.array with shape (C,) per-channel std
    """
    patch_files = glob.glob(os.path.join(cropped_root, '**', '*.npy'), recursive=True)
    total_pixels = 0
    sum_c = None
    sumsq_c = None
    min_c = None
    max_c = None
    channels = None
    patches_processed = 0

    for patch_path in patch_files:
        arr = np.load(patch_path)
        if arr.ndim != 3:
            continue
        C, H, W = arr.shape
        if channels is None:
            channels = C
        elif C != channels:
            continue
        arr = arr.astype(np.float64) * scale_factor
        n_pix = H * W
        flat = arr.reshape(C, -1)
        flat_sum = flat.sum(axis=1)
        flat_sumsq = (flat ** 2).sum(axis=1)
        flat_min = flat.min(axis=1)
        flat_max = flat.max(axis=1)
        if sum_c is None:
            sum_c = flat_sum
            sumsq_c = flat_sumsq
            min_c = flat_min
            max_c = flat_max
        else:
            sum_c += flat_sum
            sumsq_c += flat_sumsq
            min_c = np.minimum(min_c, flat_min)
            max_c = np.maximum(max_c, flat_max)
        total_pixels += n_pix
        patches_processed += 1
        if max_patches is not None and patches_processed >= max_patches:
            break

    if total_pixels == 0:
        print("No patch pixels processed, cannot compute mean/std/min/max")
        return None, None, None, None

    mean = sum_c / total_pixels
    var = (sumsq_c / total_pixels) - (mean ** 2)
    var = np.maximum(var, 0)
    std = np.sqrt(var)
    return mean, std, min_c, max_c

def main():
    root_path = '/home/weikang/MineNetCDV3/MNCDV3/MineNetCDV3_Processed_NEW'
    patch_size, step_size = 224, 112
    save_path = f'/home/weikang/MineNetCDV3/MineNetCDV3_Bitemporal_Cropped_Size{patch_size}_Step{step_size}'
    if not os.path.exists(root_path):
        print(f"Root path '{root_path}' does not exist")
        return {}

    total_patches = 0
    domain_stats = get_domain_image_stats(root_path)

    

    # cropped_root = 'MNCDV3_Bitemporal_Cropped_Size256_Step128_2788Samples'
    # # Set your scale factor here (e.g., 1e-4)
    # scale_factor = 1e-4  # Change to 1e-4 if needed
    # mean_c, std_c, min_c, max_c = compute_mean_std_cropped(cropped_root, scale_factor=scale_factor)
    # if mean_c is not None:
    #     print("Cropped dataset per-channel mean:", mean_c)
    #     print("Cropped dataset per-channel std :", std_c)
    #     print("Cropped dataset per-channel min :", min_c)
    #     print("Cropped dataset per-channel max :", max_c)

    # # Calculate Mean, Std, Min, Max from Original Dataset
    # mean, std, min_v, max_v = compute_mean_std(domain_stats, scale_factor=scale_factor)
    # if mean is not None:
    #     print("Original dataset per-channel mean:", mean)
    #     print("Original dataset per-channel std :", std)
    #     print("Original dataset per-channel min :", min_v)
    #     print("Original dataset per-channel max :", max_v)

    # Cropping Images and Labels

    patches=[]
    domains=[]

    for domain, stats in domain_stats.items():
        pre_image, post_image, pre_label, post_label = read_image_per_domain(stats)

        assert pre_image.shape[1:]==post_image.shape[1:]==pre_label.shape[1:]==post_label.shape[1:], f"Mismatch in data shapes for domain {domain}"
        # print(pre_image.shape, post_image.shape, pre_label.shape, post_label.shape)
        if pre_image is not None:
            domain_save_path_pre_img = os.path.join(save_path, domain, 'pre', 'image')
            os.makedirs(domain_save_path_pre_img, exist_ok=True)
            n_pre_img=crop_and_save_patches(domain_save_path_pre_img, 'pre', patch_size, step_size, pre_image)
        if post_image is not None:
            domain_save_path_post_img = os.path.join(save_path, domain, 'post', 'image')
            os.makedirs(domain_save_path_post_img, exist_ok=True)
            n_post_img=crop_and_save_patches(domain_save_path_post_img, 'post', patch_size, step_size, post_image)
        if pre_label is not None:
            domain_save_path_pre_label = os.path.join(save_path, domain, 'pre', 'label')
            os.makedirs(domain_save_path_pre_label, exist_ok=True)
            n_pre_label=crop_and_save_patches_rgb_label(domain_save_path_pre_label, 'pre_label', patch_size, step_size, pre_label)
        if post_label is not None:
            domain_save_path_post_label = os.path.join(save_path, domain, 'post', 'label')
            os.makedirs(domain_save_path_post_label, exist_ok=True)
            n_post_label=crop_and_save_patches_rgb_label(domain_save_path_post_label, 'post_label', patch_size, step_size, post_label)
        assert n_pre_img == n_post_img == n_pre_label == n_post_label, f"Mismatch in number of patches for domain {domain}"
        print(f"Domain {domain} shape of {pre_image.shape[1:]} processed successfully with {n_pre_img} patches.")
        total_patches += n_pre_img
        patches.append(n_pre_img)
        domains.append(domain)

    print(dict(zip(domains, patches)))
    print(f"Total patches processed across all domains: {total_patches}")

if __name__ == "__main__":
    main()
        # compute dataset mean/std across pre+post images (quick check)
    
# Stats with 1.0 Factor
# Cropped dataset per-channel mean: [1457.13412554 1481.48270723 1671.32073815 1850.16953519 2228.53605471
#  2415.18833882 2362.43300008 2451.79276064 1835.16600245]
# Cropped dataset per-channel std : [ 672.36039694  806.43486145 1099.30002642 1033.05903242  872.52339854
#   867.52872955  847.11228598 1122.82571435 1066.15523864]
# Cropped dataset per-channel min : [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Cropped dataset per-channel max : [12238. 16399. 17502. 18151. 16399. 23133. 24306. 17170. 26611.]
# Original dataset per-channel mean: [1431.45615262 1453.31232956 1640.20365717 1821.08467658 2204.21957135
#  2392.60436453 2342.00455882 2430.52003727 1813.75305148]
# Original dataset per-channel std : [ 670.04630003  801.1685072  1092.46075019 1028.58807501  874.03997329
#   872.56935658  851.73307262 1133.71869472 1069.93736439]
# Original dataset per-channel min : [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Original dataset per-channel max : [12238. 16399. 17502. 18151. 16399. 23133. 24306. 17170. 26611.]

# Stats with 1e-4 Factor
# Cropped dataset per-channel mean: [0.14571341 0.14814827 0.16713207 0.18501695 0.22285361 0.24151883
#  0.2362433  0.24517928 0.1835166 ]
# Cropped dataset per-channel std : [0.06723604 0.08064349 0.10993    0.1033059  0.08725234 0.08675287
#  0.08471123 0.11228257 0.10661552]
# Cropped dataset per-channel min : [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Cropped dataset per-channel max : [1.2238 1.6399 1.7502 1.8151 1.6399 2.3133 2.4306 1.717  2.6611]
# Original dataset per-channel mean: [0.14314562 0.14533123 0.16402037 0.18210847 0.22042196 0.23926044
#  0.23420046 0.243052   0.18137531]
# Original dataset per-channel std : [0.06700463 0.08011685 0.10924608 0.10285881 0.087404   0.08725694
#  0.08517331 0.11337187 0.10699374]
# Original dataset per-channel min : [0. 0. 0. 0. 0. 0. 0. 0. 0.]
# Original dataset per-channel max : [1.2238 1.6399 1.7502 1.8151 1.6399 2.3133 2.4306 1.717  2.6611]