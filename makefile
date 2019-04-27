package:
	mvn clean package -Dmaven.test.skip=true

compile:
	mvn clean compile -U -Dmaven.test.skip=true