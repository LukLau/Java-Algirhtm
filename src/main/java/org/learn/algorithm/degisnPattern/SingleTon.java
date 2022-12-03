package org.learn.algorithm.degisnPattern;

import org.springframework.beans.factory.BeanClassLoaderAware;

import javax.annotation.Resource;

/**
 * @author luk
 * @date 2021/7/25
 */
public class SingleTon {
    private static volatile SingleTon instance = null;

    private SingleTon() {

    }

    public static SingleTon getInstance() {
        if (instance == null) {
            synchronized (SingleTon.class) {
                if (instance == null) {
                    instance = new SingleTon();
                }
            }
        }
        return instance;
    }
}
