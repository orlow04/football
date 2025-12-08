FROM nvcr.io/nvidia/pytorch:22.12-py3

ENV DEBIAN_FRONTEND=noninteractive

# 1. Instala dependências do sistema (Mantive as suas, que estão corretas)
RUN apt-get update && apt-get --no-install-recommends install -yq \
    git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
    python3-pip

# 2. Instala dependências Python Básicas
RUN python3 -m pip install psutil
# Nota: wheel e setuptools já costumam vir atualizados nessa imagem da NVIDIA, 
# mas não faz mal garantir.
RUN python3 -m pip install wheel==0.37.0 setuptools==57.5.0

# 3. [CRÍTICO] Instala Ray, RLlib e Gymnasium
# O código de treino que fizemos usa ray e gymnasium. 
# O gfootball original usa 'gym' antigo, então instalamos o shim de compatibilidade.
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install "ray[rllib]" gymnasium

# 4. Instala o GFootball (assumindo que você está na pasta raiz do repo)
COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .

# 5. Correção do OpenCV (Mantida a sua lógica)
RUN python3 -m pip install opencv-fixer
RUN python3 -c "from opencv_fixer import AutoFix; AutoFix()"

# 6. Define diretório de trabalho
WORKDIR '/gfootball'