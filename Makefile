run:
	sudo docker compose -f infra/compose.yaml up --remove-orphans
build:
	sudo docker compose -f infra/compose.yaml up --build -d --remove-orphans
down:
	sudo docker compose -f infra/compose.yaml down --remove-orphans
