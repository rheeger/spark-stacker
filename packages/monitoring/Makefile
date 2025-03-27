.PHONY: start-monitoring stop-monitoring clean-monitoring

start-monitoring:
	cd docker && docker-compose up -d

stop-monitoring:
	cd docker && docker-compose down

clean-monitoring:
	cd docker && docker-compose down -v

# Frontend development
.PHONY: install build start

install:
	yarn install

build:
	yarn nx build monitoring

start:
	yarn nx serve monitoring
