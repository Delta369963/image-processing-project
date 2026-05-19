import { useState } from "react"

import axios from "axios"

import { motion } from "framer-motion"

function App() {

  const [selectedImage, setSelectedImage] =
    useState(null)

  const [prediction, setPrediction] =
    useState(null)

  const [loading, setLoading] =
    useState(false)

  // =====================================================
  // HANDLE IMAGE UPLOAD
  // =====================================================

  const handleImageUpload = async (event) => {

    const file = event.target.files[0]

    if (!file) return

    // IMAGE PREVIEW

    const imageURL =
      URL.createObjectURL(file)

    setSelectedImage(imageURL)

    // FORM DATA

    const formData = new FormData()

    formData.append("file", file)

    try {

      setLoading(true)

      const response =
        await axios.post(

          "http://127.0.0.1:8000/predict",

          formData,

          {
            headers: {
              "Content-Type":
                "multipart/form-data"
            }
          }
        )

      console.log(response.data)

      setPrediction(response.data)

    } catch (error) {

      console.error(error)

      alert(
        "Prediction failed"
      )

    } finally {

      setLoading(false)
    }
  }

  return (

    <div className="min-h-screen bg-slate-950 text-white overflow-hidden">

      {/* BACKGROUND */}

      <div className="absolute inset-0">

        <div className="absolute top-[-100px] left-[-100px] w-[400px] h-[400px] bg-cyan-500/20 rounded-full blur-3xl" />

        <div className="absolute bottom-[-100px] right-[-100px] w-[400px] h-[400px] bg-blue-700/20 rounded-full blur-3xl" />

      </div>

      {/* MAIN */}

      <div className="relative z-10 px-10 py-12">

        {/* HEADER */}

        <motion.div
          initial={{ opacity: 0, y: -30 }}
          animate={{ opacity: 1, y: 0 }}
        >

          <h1 className="text-6xl font-extrabold">

            Wall Crack Detection AI

          </h1>

          <p className="text-slate-400 mt-4 text-lg">

            CNN powered structural crack analysis.

          </p>

        </motion.div>

        {/* GRID */}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">

          {/* LEFT PANEL */}

          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            className="
              bg-white/5
              border border-white/10
              rounded-3xl
              p-8
              backdrop-blur-xl
            "
          >

            <h2 className="text-2xl font-semibold mb-6">

              Upload Wall Image

            </h2>

            <div
              className="
                border-2 border-dashed
                border-cyan-400/40
                rounded-2xl
                h-[350px]
                flex flex-col
                items-center
                justify-center
                text-center
                hover:border-cyan-300
                transition
                cursor-pointer
                overflow-hidden
                relative
              "
              onClick={() => {

                document
                  .getElementById(
                    "fileUpload"
                  )
                  .click()
              }}
            >

              {

                selectedImage ? (

                  <img
                    src={selectedImage}
                    alt="preview"
                    className="
                      w-full
                      h-full
                      object-cover
                    "
                  />

                ) : (

                  <>

                    <div className="text-7xl mb-4">

                      🖼️

                    </div>

                    <p className="text-xl font-medium">

                      Drag & Drop Image

                    </p>

                    <p className="text-slate-400 mt-2">

                      Click to browse

                    </p>

                  </>
                )
              }

              <input
                id="fileUpload"
                type="file"
                className="hidden"
                onChange={
                  handleImageUpload
                }
              />

            </div>

          </motion.div>

          {/* RIGHT PANEL */}

          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={{ opacity: 1, x: 0 }}
            className="
              bg-white/5
              border border-white/10
              rounded-3xl
              p-8
              backdrop-blur-xl
            "
          >

            <h2 className="text-2xl font-semibold">

              Prediction Results

            </h2>

            {

              loading ? (

                <div className="mt-10 text-cyan-300">

                  Analysing image...

                </div>

              ) : prediction ? (

                <>

                  {/* PREDICTION */}

                  <div className="mt-8">

                    <span className="
                      px-4 py-2
                      rounded-full
                      bg-cyan-500/20
                      text-cyan-300
                    ">

                      {
                        prediction.prediction
                      }

                    </span>

                  </div>

                  {/* CONFIDENCE */}

                  <div className="mt-10">

                    <div className="flex justify-between mb-2">

                      <span>
                        Confidence
                      </span>

                      <span>
                        {
                          prediction.confidence
                        }%
                      </span>

                    </div>

                    <div className="
                      w-full
                      h-4
                      bg-slate-800
                      rounded-full
                      overflow-hidden
                    ">

                      <div
                        className="
                          h-full
                          bg-cyan-400
                        "
                        style={{
                          width:
                          `${prediction.confidence}%`
                        }}
                      />

                    </div>

                  </div>

                </>

              ) : (

                <div className="
                  mt-10
                  text-slate-400
                ">

                  Upload an image to begin.

                </div>
              )
            }

          </motion.div>

        </div>

        {/* HEATMAP */}

        {

          prediction && (

            <motion.div
              initial={{
                opacity: 0,
                y: 40
              }}
              animate={{
                opacity: 1,
                y: 0
              }}
              className="
                mt-10
                bg-white/5
                border border-white/10
                rounded-3xl
                p-8
                backdrop-blur-xl
              "
            >

              <h2 className="text-2xl font-semibold mb-6">

                AI Attention Heatmap

              </h2>

              <img
                src={`http://127.0.0.1:8000/${prediction.heatmap_path}`}
                alt="heatmap"
                className="
                  rounded-2xl
                  w-full
                "
              />

            </motion.div>
          )
        }

      </div>

    </div>
  )
}

export default App