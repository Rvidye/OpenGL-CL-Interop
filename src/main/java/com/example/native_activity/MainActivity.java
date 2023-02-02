package com.example.native_activity;

import androidx.appcompat.app.AppCompatActivity;

import android.app.NativeActivity;
import android.content.Intent;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashSet;
import java.util.Set;

public class MainActivity extends AppCompatActivity {

    static
    {

        System.loadLibrary("lzma");
        System.loadLibrary("ahgc");
        System.loadLibrary("ahgc1.1");
        System.loadLibrary("ahgc1.2");
        System.loadLibrary("ahgm4");
        System.loadLibrary("ahgm2");
        System.loadLibrary("ahgm2.1");
        System.loadLibrary("ahgm3");
        System.loadLibrary("ahgm");
        System.loadLibrary("vmhm1.0");
        System.loadLibrary("vmhm1.1");
        System.loadLibrary("vmhm1.2");
        System.loadLibrary("vmhm1.3");
        System.loadLibrary("vmhm1.4");
        System.loadLibrary("vmhm1.5");
        System.loadLibrary("ahgcv2");
        System.loadLibrary("ion");
        System.loadLibrary("foo");
        //System.loadLibrary("SurfaceFlingerProp");
        //System.loadLibrary("openjdkjvmti");
        System.loadLibrary("native-activity");
        System.loadLibrary("GLES_mali");
        //System.loadLibrary("EGL");
        //System.loadLibrary("GLESv3");

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        try {
            Set<String> libs = new HashSet<String>();
            String mapsFile = "/proc/" + android.os.Process.myPid() + "/maps";
            BufferedReader reader = new BufferedReader(new FileReader(mapsFile));
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.endsWith(".so")) {
                    int n = line.lastIndexOf(" ");
                    libs.add(line.substring(n + 1));
                }
            }
            Log.d("Ldd", libs.size() + " libraries:");
            for (String lib : libs) {
                Log.d("Ldd", lib);
            }
        } catch (FileNotFoundException e) {
            // Do some error handling...
        } catch (IOException e) {
            // Do some error handling...
        }

        try {

            File nativedir = new File(getApplicationInfo().nativeLibraryDir);
            final String[] nl = nativedir.list();

            for(String s : nl)
                Log.i("Libraries :: ",s);

            Intent intent = new Intent(getApplicationContext(), NativeActivity.class);
            startActivity(intent);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }
     }
}