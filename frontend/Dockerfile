FROM node:20-slim

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install

COPY . ./

ARG VITE_API_BASE
ARG VITE_IMAGE_BASE_URL
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_IMAGE_BASE_URL=$VITE_IMAGE_BASE_URL

RUN npm run build

RUN npm install -g serve

ENV PORT=4200
CMD ["sh", "-c", "serve -s dist -l $PORT"]