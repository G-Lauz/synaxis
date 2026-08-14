# syntax=docker/dockerfile:1
FROM python:3.11 AS base

LABEL maintainer.name="Gabriel Lauzier"
LABEL maintainer.email="gabriel.lauzier@usherbrooke.ca"

ARG USER=synaxis-dev
ARG UID=10042
ARG GID=10042

ENV VIRTUAL_ENV=/opt/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV HOME=/home/${USER}
# prevent Python from writing .pyc and __pycache__/
# This keeps the container clean and mounted source tree cleaner
ENV PYTHONDONTWRITEBYTECODE=1
# write stdout and stderr immediately
ENV PYTHONUNBUFFERED=1

# Use bash for better syntax support in RUN commands
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install packages listed in packages file
COPY ./.devcontainer/packages /tmp/packages
RUN set -eux \
    && apt-get update \
    && if [[ -s /tmp/packages ]]; then \
        sed 's/\r$//' /tmp/packages | xargs -r apt-get install -y --no-install-recommends; \
    fi \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /tmp/packages

# Keep Python dependency installation cached when only source files change.
COPY requirements.txt /tmp/requirements.txt
RUN set -eux \
    && python -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

# Add privileged user
RUN set -eux \
    && groupadd --gid "${GID}" "${USER}" \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash "${USER}" \
    && printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${USER}" > "/etc/sudoers.d/${USER}" \
    && chmod 0440 "/etc/sudoers.d/${USER}"

# Setup .bashrc convenience for the user
COPY .devcontainer/bashrc /tmp/bashrc
RUN install --owner="${UID}" --group="${GID}" --mode=0644 /tmp/bashrc "/home/${USER}/.bashrc" \
    && rm -f /tmp/bashrc

WORKDIR /synaxis
COPY . /synaxis

# Install the project in editable mode, then make the image usable by the dev user.
RUN set -eux; \
    python -m pip install --no-cache-dir --editable /synaxis; \
    chown -R "${UID}:${GID}" /synaxis "${VIRTUAL_ENV}" "/home/${USER}"

USER ${USER}
RUN git config --global --add safe.directory /synaxis
WORKDIR /synaxis

CMD ["/bin/bash"]
