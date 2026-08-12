# 1. リポジトリのクローン
!git clone https://github.com/YiyuanLinXX/SAM-CLIP.git
%cd SAM-CLIP

# 2. 必要なライブラリのインストール
!pip install -q git+https://github.com/facebookresearch/segment-anything.git
!pip install -q ftfy regex tqdm
!pip install -q git+https://github.com/openai/CLIP.git

# 入力用・出力用フォルダの作成
!mkdir -p input_images output_masks

!wget -q -O input_images/test_image.jpg "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/dog.jpg"
!mkdir -p ckpt

!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
!mv sam_vit_b_01ec64.pth ckpt/checkpoint_best.pth