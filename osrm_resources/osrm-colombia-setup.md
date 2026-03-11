# OSRM Colombia Server — Setup Instructions

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

---

## Step 1 — Get the file

You should have received a single zip file: **`osrm-colombia-setup.zip`** (~651 MB).

---

## Step 2 — Extract it

Create a folder (e.g. `~/osrm`) and extract the zip there:

```bash
mkdir ~/osrm
cd ~/osrm
unzip /path/to/osrm-colombia-setup.zip
```

After extracting, the structure should look like:

```
osrm/
├── docker-compose.yml
└── osrm_data/
    └── colombia/
        ├── colombia-latest.osrm.cell_metrics
        ├── colombia-latest.osrm.ebg
        └── ... (many more .osrm.* files)
```

---

## Step 3 — Start the server

```bash
cd ~/osrm
docker compose up -d
```

Docker will pull the OSRM image on first run (~200 MB) and start the server in the background.

---

## Step 4 — Verify it's running

```bash
docker ps --filter "name=osrm-colombia"
```

You should see the container listed with status `Up`. You can also test it:

```bash
curl "http://localhost:5050/route/v1/driving/-74.08,4.71;-75.59,6.24?overview=false"
```

A JSON response with route data means everything is working. The server is available at **`http://localhost:5050`**.

---

## Stopping / restarting

```bash
# Stop
docker compose down

# Restart
docker compose up -d
```

Since `restart: unless-stopped` is set, the server will **automatically restart** after reboots as long as Docker Desktop is running.
