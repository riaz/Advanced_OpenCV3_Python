package com.tangibleminds.emotions;

import java.io.IOException;
import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PreviewCallback;
import android.view.SurfaceHolder;
import android.view.SurfaceView;

public class Preview  extends SurfaceView implements SurfaceHolder.Callback {
	  private static final String TAG = "Preview";

	  SurfaceHolder mHolder;
	  public Camera camera;

	  Preview(Context context) {
	    super(context);

	    // Install a SurfaceHolder.Callback so we get notified when the
	    // underlying surface is created and destroyed.
	    mHolder = getHolder();  // <4>
	    mHolder.addCallback(this);  // <5>
	    mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
	  }

	  // Called once the holder is ready
	  public void surfaceCreated(SurfaceHolder holder) {
	    // The Surface has been created, acquire the camera and tell it where
	    // to draw.		 

		  camera = Camera.open();
		  
	    try {	    	
	      camera.setPreviewDisplay(holder);
	      camera.setPreviewCallback(new PreviewCallback() { // <10>
	        // Called for each frame previewed
	        public void onPreviewFrame(byte[] data, Camera camera) {
	           Preview.this.invalidate();
	        }
	      });
	    } catch (IOException e) {
	      e.printStackTrace();
	    }
	  }

	  // Called when the holder is destroyed
	  public void surfaceDestroyed(SurfaceHolder holder) {
	    camera.stopPreview();
	    camera.setPreviewCallback(null);
	    camera.release();
	    camera = null;
	  }

	  // Called when holder has changed
	  public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
	    camera.startPreview();
	  }
}
