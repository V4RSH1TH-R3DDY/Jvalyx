# syntax=docker/dockerfile:1

# ---- build ----------------------------------------------------------------
# Vite inlines `import.meta.env.VITE_*` at build time, so the API base URL and the
# optional map keys must be supplied as build args, not runtime env vars. They end up
# in the shipped JS bundle — treat anything passed here as public (see
# docs/cross-layer-findings.md L2 on the FIRMS key).
FROM node:24-alpine AS build

WORKDIR /app

# Install against the lockfile first so dependency layers cache across source edits.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Points the browser at the published backend port, not at a container hostname:
# these calls are made by the user's browser, which is outside the compose network.
ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_FIRMS_MAP_KEY=
ARG VITE_CARTO_API_KEY=
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL \
    VITE_FIRMS_MAP_KEY=$VITE_FIRMS_MAP_KEY \
    VITE_CARTO_API_KEY=$VITE_CARTO_API_KEY

RUN npm run build

# ---- serve ----------------------------------------------------------------
FROM nginx:1.27-alpine AS runtime

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD wget -q --spider http://127.0.0.1/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
