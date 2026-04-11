FROM node:20-slim

WORKDIR /usr/src/app

COPY package*.json ./
RUN npm install

COPY . ./
RUN npm run build

RUN npm install -g serve

ENV PORT=4200
CMD ["sh", "-c", "serve -s dist -l $PORT"]