# to build this dockerfile, run below:
# docker build -t $USER/cuda:latest .

# for other distros, go to https://hub.docker.com/r/nvidia/cuda
FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04
LABEL author="syool"

# non-interactive installation of gdb
RUN export DEBIAN_FRONTEND=noninteractive
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime

# install preliminaries
# for opencv, add following package: libgl1-mesa-glx
RUN apt update && apt upgrade -y
RUN apt install git curl wget zip zsh micro gdb python3 python3-pip libgl1-mesa-glx build-essential -y

# upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# install pytorch
RUN pip3 install vllm accelerate

# install packages for RAG orchestration
RUN pip3 install -U langchain langgraph
RUN pip3 install llama-index haystack

# install packages for data ingestion & parsing
RUN pip3 install pypdf unstructured python-docx firecrawl-py tiktoken

# install packages for vector db & embeddings
RUN pip3 install sentence-transformers faiss-gpu chromadb qdrant-client

# install packages for evaluation
RUN pip3 install ragas

# install dependencies
RUN pip3 install scikit-learn opencv-python \
    matplotlib plotly tqdm ipykernel lightning \
	wandb streamlit

# install packages for WTA
RUN pip3 install gymnasium stable-baselines3

# install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

# install zsh theme "spaceship"
RUN git clone https://github.com/spaceship-prompt/spaceship-prompt.git "$HOME/.oh-my-zsh/custom/themes/spaceship-prompt" --depth=1
RUN ln -s "$HOME/.oh-my-zsh/custom/themes/spaceship-prompt/spaceship.zsh-theme" "$HOME/.oh-my-zsh/custom/themes/spaceship.zsh-theme"

# zsh settings
RUN echo "export TERM=xterm-256color\n\
export ZSH=\"$HOME/.oh-my-zsh\"\n\
export PATH=\"$HOME/.local/bin:$PATH\"\n\
ZSH_THEME=\"spaceship\"\n\
zmodload zsh/nearcolor\n\
SPACESHIP_CHAR_COLOR_SUCCESS=\"#5fff00\"\n\
SPACESHIP_CHAR_COLOR_FAILURE=\"#ff0000\"\n\
SPACESHIP_USER_SUFFIX=\"\"\n\
SPACESHIP_USER_COLOR=\"#ffffff\"\n\
SPACESHIP_HOST_SHOW=\"always\"\n\
SPACESHIP_HOST_PREFIX=\"@\"\n\
SPACESHIP_HOST_SUFFIX=\" \"\n\
SPACESHIP_HOST_COLOR=\"#ffffff\"\n\
SPACESHIP_DIR_PREFIX=\"\"\n\
SPACESHIP_DIR_TRUNC=\"0\"\n\
SPACESHIP_DIR_COLOR=\"#40826d\"\n\
SPACESHIP_GIT_PREFIX=\"\"\n\
SPACESHIP_GIT_BRANCH_COLOR=\"#6c6c6c\"\n\
SPACESHIP_GIT_STATUS_PREFIX=\" \"\n\
SPACESHIP_GIT_STATUS_SUFFIX=\"\"\n\
SPACESHIP_GIT_STATUS_COLOR=\"#00d7ff\"\n\
SPACESHIP_EXEC_TIME_PREFIX=\"\"\n\
SPACESHIP_EXEC_TIME_COLOR=\"#ffaf00\"\n\
SPACESHIP_CONDA_COLOR=\"#11a3a2\"\n\
SPACESHIP_CONDA_PREFIX=\"\"\n\
SPACESHIP_DOCKER_PREFIX=\"\"\n\
SPACESHIP_PROMPT_ORDER=(\n\
	time\n\
	user\n\
	host\n\
	dir\n\
	git\n\
	docker\n\
	exec_time\n\
	line_sep\n\
	jobs\n\
	exit_code\n\
	conda\n\
	char\n\
)\n\
plugins=(\n\
	git\n\
)\n\
source \$ZSH/oh-my-zsh.sh" > ~/.zshrc

# micro editor settings
RUN mkdir -p ~/.config/micro
RUN echo "{\"colorscheme\": \"bubblegum\"}" > ~/.config/micro/settings.json

# install miniconda3
# RUN sh -c "$(curl -o ~/miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh)"
# RUN sh ~/miniconda3.sh -b -p && \
#     rm ~/miniconda3.sh
# RUN ~/miniconda3/bin/conda clean -tipsy && \
#     ln -s ~/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
#     echo ". ~/miniconda3/etc/profile.d/conda.sh\n \
#           conda activate base" >> ~/.zshrc

WORKDIR /root

CMD [ "/bin/zsh" ]
