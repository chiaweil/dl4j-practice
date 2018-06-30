package org.deeplearning4j.examples.test;

/**
 * Override public void testFunc(String input) during declaration of function
 */
public class SampleClass
{
    public SampleClass(String init)
    {
        System.out.println("init SampleClass");
    }

    public void testFunc(String input)
    {
        System.out.println("Default testFunc()");
    }

}
