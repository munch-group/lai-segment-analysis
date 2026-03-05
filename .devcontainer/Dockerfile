FROM mcr.microsoft.com/devcontainers/base:jammy

# ARG PIXI_VERSION=v0.42.1

#RUN apt-get update && apt-get install -y curl

#RUN curl -fsSL https://pixi.sh/install.sh | bash
# RUN wget -qO- https://pixi.sh/install.sh | sh

# RUN curl -L -o /usr/local/bin/pixi -fsSL --compressed "https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-$(uname -m)-unknown-linux-musl" \
    # && chmod +x /usr/local/bin/pixi \
    # && pixi info
# && chmod +x /usr/local/bin/pixi  && pixi info

# set some user and workdir settings to work nicely with vscode
USER vscode
WORKDIR /home/vscode

RUN wget -qO- https://pixi.sh/install.sh | /bin/bash
# RUN wget -qO- https://pixi.sh/install.sh | sh


# RUN echo 'eval "$(pixi completion -s bash)"' >> /home/vscode/.bashrc
# RUN echo 'eval "$(pixi shell-hook -s bash)"' >> /home/vscode/.bashrc

# # RUN --mount=type=secret,id=prefix_dev_token,uid=1000 \
#     test -s /run/secrets/prefix_dev_token \
#     && pixi auth login --token "$(cat /run/secrets/prefix_dev_token)" https://repo.prefix.dev

