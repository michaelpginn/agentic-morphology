FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential sudo nodejs npm && \
    pip install pyfoma==1.1.1 && \
    npm install -g @anthropic-ai/claude-code@2.1.202 && \
    apt-get purge -y build-essential && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*


RUN useradd -m agent && mkdir /workspace && chown agent /workspace

COPY grader/ /opt/grader/
RUN chmod -R 700 /opt/grader && \
    echo "agent ALL=(root) NOPASSWD: /opt/grader/grade" >> /etc/sudoers

COPY skills/ /workspace/skills/
RUN chmod -R a+rX /workspace/skills

USER agent
WORKDIR /workspace
