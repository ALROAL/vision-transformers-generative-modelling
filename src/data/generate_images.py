from torchvision.utils import save_image
from src import _PATH_DATA, _PATH_MODELS, _PROJECT_ROOT
from src.models.models import CViTVAE, ConvCVAE


if __name__ == "__main__":
    model = CViTVAE.load_from_checkpoint("/work3/s164564/Vision-transformers-for-generative-modeling/models/CViTVAE2022-04-29-1735/CViTVAE-epoch=174.ckpt")
    model.eval()
    for i in range(10):
        img = model.forward_2(torch.zeros(1),1000)
        for j,im in enumerate(img):
            save_image(im,"../data/CViTVAE/{}.png".format(1000*i+j))

    model = ConvCVAE().load_from_checkpoint("F:\Vision-transformers-for-generative-modeling\models\ConvCVAE2022-04-30-1854\ConvCVAE-epoch=349.ckpt")
    model.eval()
    for i in range(10):
        img = model.forward_2(torch.zeros(1),1000)
        for j,im in enumerate(img):
            save_image(im,"../data/CViTVAE/{}.png".format(1000*i+j))