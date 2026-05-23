COMPOSE_FILE=infra/compose.yaml
build:
		sudo docker compose -f $(COMPOSE_FILE) up --build $(s)
down:
		sudo docker compose -f $(COMPOSE_FILE) down -v $(s)
ps:
		sudo docker ps
bash:
		sudo docker exec -it $(id) bash
wipe:
		sudo docker compose -f $(COMPOSE_FILE) down -v --remove-orphans $(s)
rebuild:
		$(MAKE) wipe
		$(MAKE) build
