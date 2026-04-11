FROM node:20-slim

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install

COPY . ./
ARG VITE_API_BASE
ENV VITE_API_BASE=$VITE_API_BASE

ARG VITE_IMAGE_BASE
ENV VITE_IMAGE_BASE=$VITE_IMAGE_BASE

RUN npm run build

RUN npm install -g serve

ENV PORT=4200
CMD ["sh", "-c", "serve -s dist -l $PORT"]