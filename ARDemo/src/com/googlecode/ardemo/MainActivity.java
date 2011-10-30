package com.googlecode.ardemo;

import java.util.List;

import org.opencv.core.Size;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.Window;
import android.view.WindowManager;

public class MainActivity extends Activity {
    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "onCreate");

        super.onCreate(savedInstanceState);

        // Without title we will have more free space
        if (!requestWindowFeature(Window.FEATURE_NO_TITLE)) {
            Log.i(TAG, "Can't request NO_TITLE window feature"); 
        }
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        cameraView = new GlCameraView(this);
        cameraView.setFrameProcessor(circles);

        setContentView(cameraView);
    }

    @Override
    protected void onPause() {
        Log.i(TAG, "onPause");

        cameraView.onPause();

        super.onPause();
    }

    @Override
    protected void onResume() {
        Log.i(TAG, "onResume");

        cameraView.onResume();

        super.onResume();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "onCreateOptionsMenu");

        menu.add(0, TOGGLE_FPS, 0, R.string.toogle_fps);

        List<Size> resolutions = cameraView.getAvaibleCameraResolutions();
        if (resolutions != null) {
            SubMenu resolutionMenu = menu.addSubMenu(0, RESOLUTION_SUBMENU, 0, R.string.camera_resolution);
            int ind = 0;
            for (Size size : resolutions) {
                resolutionMenu.add(0, RESOLUTION_START + ind, 0, size.toString());
                ++ind;
            }
        }

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "onOptionsItemSelected");

        switch (item.getItemId()) {
            case TOGGLE_FPS:
                cameraView.toggleFps();
                break;

            default:
                int itemId = item.getItemId();

                if (itemId >= RESOLUTION_START) {
                    int ind = itemId - RESOLUTION_START;

                    List<Size> resolutions = cameraView.getAvaibleCameraResolutions();

                    if (resolutions != null && ind >= 0 && ind < resolutions.size()) {
                        Size resolution = resolutions.get(ind);
                        cameraView.setCameraResolution(resolution);
                    }
                }

                break;
        }

        return true;
    }

    private GlCameraView cameraView = null;

    private CirclesDetector circles = new CirclesDetector();

    private static final int TOGGLE_FPS         = Menu.FIRST;

    private static final int RESOLUTION_SUBMENU = TOGGLE_FPS + 1;
    private static final int RESOLUTION_START   = RESOLUTION_SUBMENU + 1;

    private static final String TAG = "UltraEye::MainActivity";
}