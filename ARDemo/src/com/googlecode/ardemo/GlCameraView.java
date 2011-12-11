package com.googlecode.ardemo;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import org.opencv.core.Size;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLSurfaceView;
import android.opengl.GLU;
import android.opengl.GLUtils;
import android.util.AttributeSet;
import android.util.Log;

public class GlCameraView extends GLSurfaceView {
    public GlCameraView(Context context) {
        super(context);
        init();
    }

    public GlCameraView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }

    public void toggleFps() {
        camera.toggleFps();
    }

    public List<Size> getAvaibleCameraResolutions() {
        return camera.getAvaibleCameraResolutions();
    }

    public void setCameraResolution(Size resolution) {
        camera.setCameraResolution(resolution);
    }

    public void setDetector(Detector detector) {
        camera.setDetector(detector);
    }

    @Override
    public void onPause() {
        Log.i(TAG, "onPause");

        camera.release();

        super.onPause();
    }

    @Override
    public void onResume() {
        Log.i(TAG, "onResume");

        camera.create();
        camera.resume();

        super.onResume();
    }

    private void init() {
        Log.i(TAG, "Init OpenGL");

        setEGLConfigChooser(true);
        setRenderer(renderer);
        setRenderMode(GLSurfaceView.RENDERMODE_CONTINUOUSLY);

        Log.i(TAG, "Create buffers");

        final int VERTICES_COUNT = 6; // 6 vertices for 2 triangles
        ByteBuffer vbb = ByteBuffer.allocateDirect(VERTICES_COUNT * 2 * 4);
        vbb.order(ByteOrder.nativeOrder());
        vertices = vbb.asFloatBuffer();

        ByteBuffer tbb = ByteBuffer.allocateDirect(VERTICES_COUNT * 2 * 4);
        tbb.order(ByteOrder.nativeOrder());
        texCoords = tbb.asFloatBuffer();

        vertices. put(0.0f); vertices. put(0.0f);
        texCoords.put(0.0f); texCoords.put(1.0f);
        vertices. put(0.0f); vertices. put(1.0f);
        texCoords.put(0.0f); texCoords.put(0.0f);
        vertices. put(1.0f); vertices. put(0.0f);
        texCoords.put(1.0f); texCoords.put(1.0f);

        vertices. put(1.0f); vertices. put(0.0f);
        texCoords.put(1.0f); texCoords.put(1.0f);
        vertices. put(0.0f); vertices. put(1.0f);
        texCoords.put(0.0f); texCoords.put(0.0f);
        vertices. put(1.0f); vertices. put(1.0f);
        texCoords.put(1.0f); texCoords.put(0.0f);

        vertices.position(0);
        texCoords.position(0);
    }

    private final Renderer renderer = new Renderer() {
        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            Log.i(TAG, "onSurfaceCreated");

            gl.glMatrixMode(GL10.GL_PROJECTION);
            gl.glLoadIdentity();
            GLU.gluOrtho2D(gl, 0.0f, 1.0f, 0.0f, 1.0f);
            gl.glMatrixMode(GL10.GL_MODELVIEW);

            Log.i(TAG, "Create textures");

            gl.glEnable(GL10.GL_TEXTURE_2D);
            int[] textures = new int[1];
            gl.glGenTextures(1, textures, 0);
            texId = textures[0];
            gl.glBindTexture(GL10.GL_TEXTURE_2D, texId);
            gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_MIN_FILTER, GL10.GL_LINEAR);
            gl.glTexParameterf(GL10.GL_TEXTURE_2D, GL10.GL_TEXTURE_MAG_FILTER, GL10.GL_LINEAR);

            Log.i(TAG, "Texture was created: " + texId);

            gl.glEnableClientState(GL10.GL_VERTEX_ARRAY);
            gl.glEnableClientState(GL10.GL_TEXTURE_COORD_ARRAY);
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            Log.i(TAG, "onSurfaceChanged");

            camera.updateResolution(width, height);

            gl.glViewport(0, 0, width, height);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            Log.i(TAG, "onDrawFrame");

            gl.glClear(GL10.GL_COLOR_BUFFER_BIT | GL10.GL_DEPTH_BUFFER_BIT);

            Bitmap bmp = camera.processFrame();

            if (bmp != null) {
                Log.i(TAG, "Draw texture");

                gl.glDisable(GL10.GL_DEPTH_TEST);

                GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, bmp, 0);

                gl.glVertexPointer(2, GL10.GL_FLOAT, 0, vertices);
                gl.glTexCoordPointer(2, GL10.GL_FLOAT, 0, texCoords);

                gl.glDrawArrays(GL10.GL_TRIANGLES, 0, 6);
            }
        }
    };

    private final CameraHolder camera = new CameraHolder();

    private int texId;

    private FloatBuffer vertices;
    private FloatBuffer texCoords;

    private static final String TAG = "ARDemo::GlCameraView";
}
