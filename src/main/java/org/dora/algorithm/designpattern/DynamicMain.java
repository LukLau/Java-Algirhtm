package org.dora.algorithm.designpattern;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;

interface Service {
    public int service(int paran);
}

public class DynamicMain {

}

class ServiceImpl implements Service {

    @Override
    public int service(int paran) {
        return paran;
    }
}

class DynamicInvocationHandler implements InvocationHandler {
    private Object target;

    public DynamicInvocationHandler(Object target) {
        this.target = target;
    }

    @Override
    public Object invoke(Object proxy, Method method, Object[] args) throws Throwable {
        return  method.invoke(target, args);
    }
}
