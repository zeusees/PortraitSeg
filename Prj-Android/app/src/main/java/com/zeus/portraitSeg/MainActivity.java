// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.zeus.portraitSeg;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.Spinner;

public class MainActivity extends Activity implements SurfaceHolder.Callback {
    // 初始化的一些变量
    public static final int REQUEST_CAMERA = 100;

    private NanoDetNcnn nanodetncnn = new NanoDetNcnn(); // 调用ncnn的java接口类
    private int facing = 0; // 用来记录前摄还是后摄

    private Spinner spinnerCPUGPU; // 切换CPU/GPU
    private int current_cpugpu = 0; // 记录当前内核

    private SurfaceView cameraView; // 预览界面

    // 初始化函数
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main); // 加载布局

        // 保持屏幕唤醒不锁屏
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // 绑定预览的控件，并设置格式为RGBA8888
        cameraView = (SurfaceView) findViewById(R.id.cameraview);
        cameraView.getHolder().setFormat(PixelFormat.RGBA_8888);
        cameraView.getHolder().addCallback(this);

        DisplayMetrics metric = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(metric);
        int screenW = metric.widthPixels;
        int h = 640, w = 480;

        int fixedScreenH = h * screenW / w;// 宽度不变，等比缩放的高度

        LinearLayout.LayoutParams layoutParams = (LinearLayout.LayoutParams) cameraView.getLayoutParams();
        layoutParams.width = screenW;
        layoutParams.height = fixedScreenH;
        cameraView.setLayoutParams(layoutParams);

        reload();
    }

    private void reload() {
        // 重新加载当前选定的模型到指定的内核
        boolean ret_init = nanodetncnn.loadModel(getAssets(), current_cpugpu);
        if (!ret_init) {
            Log.e("MainActivity", "nanodetncnn loadModel failed");
        }
    }

    // 这是SurfaceView需要实现的三个方法
    // surface尺寸发生改变的时候调用，如横竖屏切换
    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        nanodetncnn.setOutputWindow(holder.getSurface());
    }

    // surface创建的时候调用，一般在该方法中启动绘图的线程
    @Override
    public void surfaceCreated(SurfaceHolder holder) {
    }

    // surface被销毁的时候调用，如退出游戏画面，一般在该方法中停止绘图线程
    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
    }

    // onPause方法，app被覆盖后就把ncnn的摄像头关了就完事了
    @Override
    public void onPause() {
        super.onPause();
        nanodetncnn.closeCamera();
    }

    // onResume方法，app被覆盖恢复后把摄像头重新开起来
    @Override
    public void onResume() {
        super.onResume();
        if (ContextCompat.checkSelfPermission(getApplicationContext(), Manifest.permission.CAMERA) == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        }
        nanodetncnn.openCamera(facing);
    }
}
