# Codabench Worker Setup

This folder contains a Docker Compose setup for running a Codabench compute worker.

The worker can run on any machine. It does not need to be the same machine used to build or upload the competition bundle.

## What The Worker Does

The worker listens to a Codabench queue, receives submissions, starts a separate job container for each submission, runs the ingestion and scoring programs, and sends the results back to Codabench.

There are two Docker images involved:

```text
Worker image:
codalab/codabench-compute-worker:latest

Competition job image:
pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
```

The worker image is configured in `docker-compose.yml`.

The competition job image is configured in the competition bundle's `competition.yaml`.

## Requirements

- Docker installed
- Docker Compose installed
- Access to the Codabench broker URL
- Hidden competition data copied onto the worker machine
- NVIDIA drivers and NVIDIA Container Toolkit, if GPU execution is required

## Queue Configuration

In `competition.yaml`, Codabench should use only the queue vhost:

```yaml
queue: 2bebe1bd-da61-4e0c-8a45-232eedcf87d8
```

In the worker `.env`, use the full broker URL:

```env
BROKER_URL=pyamqp://<username>:<password>@www.codabench.org:5672/<vhost>
```

Do not put the full broker URL in `competition.yaml`.

## Data Layout

The competition code expects hidden data inside the job container at:

```text
/app/data/comp_data/
```

Codabench maps the worker host directory's `data` folder into the job container as `/app/data`.

So on the worker machine, put the hidden data at:

```text
<HOST_DIRECTORY>/data/comp_data/
```

For the local setup in this repository:

```text
HOST_DIRECTORY=/home/soroush/codabench
```

The worker container mounts that host directory at `/codabench`, so the local hidden data path is:

```text
/home/soroush/codabench/data/comp_data/
```

Expected structure:

```text
comp_data/
  OCT/
    <case_id>/
      *.png
  Opmi/
    <case_id>/
      microscope.png
  Numerical/
    <case_id>/
      <case_id>.json
```

## Setup On A New Machine

Create a worker folder:

```bash
mkdir -p ~/codabench-worker
cd ~/codabench-worker
```

Create `.env`:

```env
BROKER_URL=pyamqp://<username>:<password>@www.codabench.org:5672/<vhost>
BROKER_USE_SSL=True
HOST_DIRECTORY=/codabench
CONTAINER_ENGINE_EXECUTABLE=docker
USE_GPU=True
GPU_DEVICE=nvidia.com/gpu=all
```

Create `docker-compose.yml`:

```yaml
services:
  worker:
    image: codalab/codabench-compute-worker:latest
    container_name: compute_worker
    volumes:
      - /codabench:/codabench
      - /var/run/docker.sock:/var/run/docker.sock
    env_file:
      - .env
    restart: unless-stopped
    logging:
      options:
        max-size: 50m
        max-file: "3"
```

Create the hidden-data folder:

```bash
sudo mkdir -p /codabench/data
```

Copy the dataset so this exists:

```text
/codabench/data/comp_data/
```

Start the worker:

```bash
docker pull codalab/codabench-compute-worker:latest
docker compose up -d
docker logs -f compute_worker
```

If the system has old Docker Compose, use `docker-compose` instead:

```bash
docker-compose up -d
docker logs -f compute_worker
```

## Useful Commands

Check whether the worker is running:

```bash
docker ps
```

Watch logs:

```bash
docker logs -f compute_worker
```

Restart the worker:

```bash
docker compose restart
```

Stop the worker:

```bash
docker compose down
```

For old Docker Compose:

```bash
docker-compose restart
docker-compose down
```

## Notes

- Every worker connected to the same broker URL can receive jobs from the same queue.
- If multiple workers are used, each worker must have the hidden data at the same configured path.
- The `.env` file contains queue credentials and should not be shared publicly.
- If Docker commands fail with permission errors, run them with `sudo` or add the user to the Docker group.
- If the worker logs `No such file or directory: '/codabench/...'`, make sure the host data/work directory is mounted to `/codabench` inside the worker container.
