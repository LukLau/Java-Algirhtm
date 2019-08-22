package org.dora.algorithm.designpattern;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;

interface Service {
    int service(int param);
}

public class DynamicMain {
    public static void main(String[] args) {

        Service realService = new ServiceImpl();
        DynamicHadnler dynamicHadnler = new DynamicHadnler(realService);
        Service test = (Service) Proxy.newProxyInstance(realService.getClass().getClassLoader(), realService.getClass().getInterfaces(), dynamicHadnler);
        System.out.println(test.service(123));
    }

}

class ServiceImpl implements Service {

    @Override
    public int service(int param) {
        return param;
    }
}

class DynamicHadnler implements InvocationHandler {
    private Object target;

    public DynamicHadnler(Object target) {
        this.target = target;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        return method.invoke(target, args);
    }
}
