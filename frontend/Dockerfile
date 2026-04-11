FROM node:20-slim

WORKDIR /usr/src/app

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

RUN npm install -g serve

ENV PORT=4200
CMD ["sh", "-c", "serve -s dist -l $PORT"]