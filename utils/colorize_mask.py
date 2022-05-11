from PIL import Image
import torch
import numpy as np
#调色板      #3个一组？
#第一步，转成numpy中的array

#第二步，N x W x H --> 1 x W x H（W x H），原理就是，对图像中的每一个像素，判断它在哪一类中的得分最高，然后把像素值置为得分最高类的序号。

#第三步，转成Image，调色板着色
cityscapes_palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                      220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0,
                      70,
                      0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

camvid_palette = [128, 128, 128, 128, 0, 0, 192, 192, 128, 128, 64, 128, 60, 40, 222, 128, 128, 0, 192, 128, 128, 64,
                  64,
                  128, 64, 0, 128, 64, 64, 0, 0, 128, 192]

zero_pad = 256 * 3 - len(cityscapes_palette)
for i in range(zero_pad):
    cityscapes_palette.append(0)


# zero_pad = 256 * 3 - len(camvid_palette)
# for i in range(zero_pad):
#     camvid_palette.append(0)

#mask
def cityscapes_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')#P (8-bit pixels, mapped to any other mode using a colour palette)
    new_mask.putpalette(cityscapes_palette)

    return new_mask


def camvid_colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(camvid_palette)

    return new_mask


class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = voc_color_map(22)               #cmap=color map 颜色表
        self.cmap = torch.from_numpy(self.cmap[:n]) #cmap由numpy转化成tensor

    def __call__(self, gray_image):
        size = gray_image.shape     ## 网络output的大小
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)#生成3通道的image模板

        for label in range(0, len(self.cmap)):   # 依次遍历label的颜色表
            mask = (label == gray_image)
            #mask = gray_image[0] == label
            # gray_image[0] 是将三维的图像，以【1, 10, 10】为例，变成二维【10,10】,这个参数是外部传入，这里确保是二维单通道就行了
            # gray_image[0] == label 意思是将 gray_image[0]中为label值的元素视为true或者1，其他的元素为False 或0，得到mask的布尔图

            color_image[0][mask] = self.cmap[label][0]  ##取颜色表中为label列表(【a,b,c】)的a#color_image[0]是取三通道模板中的单通道 ，然后把mask放上去
            color_image[1][mask] = self.cmap[label][1]  #取颜色表中为label列表(【a,b,c】)的b
            color_image[2][mask] = self.cmap[label][2]  #取颜色表中为label列表(【a,b,c】)的c

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

# VOC数据集颜色对应关系与代码
def voc_color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
