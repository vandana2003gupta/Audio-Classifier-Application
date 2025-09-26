package com.example.mysoundclassification

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.task.audio.classifier.AudioClassifier
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private val TAG = "MainActivity"

    // Path to the model (make sure this file is inside app/src/main/assets/)
    private val modelPath = "1.tflite"

    // Minimum probability threshold for results
    private val probabilityThreshold: Float = 0.3f

    private lateinit var outputTextView: TextView
    private lateinit var recorderSpecsTextView: TextView

    private val REQUEST_RECORD_AUDIO = 1337

    // Executor for background classification
    private val executor = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI elements
        outputTextView = findViewById(R.id.output)
        recorderSpecsTextView = findViewById(R.id.textViewAudioRecorderSpecs)

        // Check and request microphone permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                REQUEST_RECORD_AUDIO
            )
        } else {
            startAudioClassification()
        }
    }

    private fun startAudioClassification() {
        try {
            // Load the model
            val classifier = AudioClassifier.createFromFile(this, modelPath)

            // Create input tensor
            val tensor = classifier.createInputTensorAudio()

            // Show recorder specs
            val format = classifier.requiredTensorAudioFormat
            val recorderSpecs = "Channels: ${format.channels}\nSample Rate: ${format.sampleRate}"
            runOnUiThread { recorderSpecsTextView.text = recorderSpecs }

            // Start recording
            val record = classifier.createAudioRecord()
            record.startRecording()

            // Run classification every 500ms in background
            executor.execute {
                try {
                    while (!executor.isShutdown) {
                        Thread.sleep(500)

                        // Load audio and classify
                        tensor.load(record)
                        val output = classifier.classify(tensor)

                        // Filter results above threshold
                        val filteredModelOutput = output[0].categories.filter {
                            it.score > probabilityThreshold
                        }

                        // Build result string
                        val outputStr = filteredModelOutput.sortedByDescending { it.score }
                            .joinToString(separator = "\n") {
                                "${it.label} -> ${"%.2f".format(it.score)}"
                            }

                        // Update UI safely
                        runOnUiThread {
                            outputTextView.text =
                                if (outputStr.isNotEmpty()) outputStr else "No sound detected"
                        }
                    }
                } catch (e: Exception) {
                    e.printStackTrace()
                    runOnUiThread {
                        outputTextView.text = "Error in loop: ${e.localizedMessage}"
                    }
                }
            }

        } catch (e: Exception) {
            e.printStackTrace()
            runOnUiThread {
                outputTextView.text = "Error: ${e.localizedMessage}"
            }
        }
    }

    // Handle permission result
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_RECORD_AUDIO &&
            grantResults.isNotEmpty() &&
            grantResults[0] == PackageManager.PERMISSION_GRANTED
        ) {
            startAudioClassification()
        } else {
            outputTextView.text = "Microphone permission denied."
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow() // Stop background loop
    }
}
