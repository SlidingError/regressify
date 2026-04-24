function setLoading(button, state){
    button.disabled = state
    button.innerText = state ? "Processing..." : button.dataset.label
}

document.getElementById("trainBtn").dataset.label="Train"
document.getElementById("predictBtn").dataset.label="Predict"


async function train(){

    const fileInput = document.getElementById("csvFile")
    const file = fileInput.files[0]
    const result = document.getElementById("trainResult")
    const btn = document.getElementById("trainBtn")

    if(!file){
        result.innerText="Please select a CSV file."
        return
    }

    const learningRate = parseFloat(document.getElementById("learningRate").value) || 0.01
    const iterations = parseInt(document.getElementById("iterations").value, 10) || 100
    const treeMaxDepth = parseInt(document.getElementById("treeDepth").value, 10) || 3
    const treeMinSamples = parseInt(document.getElementById("treeMin").value, 10) || 3

    const formData = new FormData()
    formData.append("file", file)
    formData.append("learning_rate", learningRate)
    formData.append("iterations", iterations)
    formData.append("max_depth", treeMaxDepth)
    formData.append("min_samples", treeMinSamples)

    try{

        setLoading(btn,true)

        const res = await fetch("http://127.0.0.1:8000/train",{
            method:"POST",
            body:formData
        })

        const data = await res.json()
        result.innerText = JSON.stringify(data,null,2)

    }catch(err){
        result.innerText="Training request failed."
    }

    setLoading(btn,false)
}


async function predict(){

    const raw = document.getElementById("inputData").value
    const result = document.getElementById("predictResult")
    const btn = document.getElementById("predictBtn")

    let json

    try{
        json = JSON.parse(raw)
    }catch(e){
        result.innerText="Invalid JSON input."
        return
    }

    try{

        setLoading(btn,true)

        const res = await fetch("http://127.0.0.1:8000/predict",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify(json)
        })

        const data = await res.json()
        result.innerText = JSON.stringify(data,null,2)

    }catch(err){
        result.innerText="Prediction request failed."
    }

    setLoading(btn,false)
}
