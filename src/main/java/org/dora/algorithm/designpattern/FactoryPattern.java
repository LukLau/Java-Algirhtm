package org.dora.algorithm.designpattern;

interface FactoryService {
    void service(String name);
}

/**
 * @author liulu
 * @date 2019-03-19
 */
public class FactoryPattern {

    public static void main(String[] args) {
        FactoryPattern factoryPattern = new FactoryPattern();
        FactoryService factoryService = factoryPattern.product(1);
        factoryService.service("test");
    }

    public FactoryService product(int type) {
        if (type == 0) {
            return new SmsFactoryService();
        }
        if (type == 1) {
            return new MailFactoryService();
        }
        return new MailFactoryService();
    }
}

class SmsFactoryService implements FactoryService {

    @Override
    public void service(String name) {
        System.out.println(name);
    }
}

class MailFactoryService implements FactoryService {

    @Override
    public void service(String name) {
        System.out.println(name);
    }
}