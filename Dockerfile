FROM python:3.10.15

RUN apt update && apt install sysstat -y

WORKDIR /app

COPY setup.py setup.py
COPY stable_baselines3/version.txt stable_baselines3/version.txt

RUN pip install -e .[docs,tests,extra]
RUN pip install wandb
RUN pip install "gymnasium[box2d]"
RUN pip install "gymnasium[mujoco]"
RUN pip install ray

RUN git clone https://github.com/eilab-gt/NovGrid.git
RUN cd NovGrid && pip install -e .
RUN pip install git+https://github.com/google-research/realworldrl_suite.git

COPY . .

ENTRYPOINT ["bash", "-c", "\
    ray start --head \
    && python experiment.py \
"]