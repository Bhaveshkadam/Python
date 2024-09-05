FROM postgres:14

LABEL authors="Bhavesh Kadam"

RUN apt-get update && \
    apt-get install -y postgresql-14-pgvector && \
    apt-get clean

EXPOSE 5432
