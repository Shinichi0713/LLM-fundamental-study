# Colab上で
!git clone https://github.com/jiny1623/DynamicER.git
%cd DynamicER
!pip install -e .  # もしsetup.pyがあれば
# Cythonのビルド
!python setup.py build_ext --inplace

!pip install llama-index llama-index-embeddings-huggingface