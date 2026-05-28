
function App () {
  return (
    <div> 
      
      <h1> CRAM - Cancer Risk Assessment Model </h1> 

      <hr></hr>

      <p> Welcome to the Cancer Risk Assesment Model, this is a tool 
        designed to help medical professionals identify cancer risk in 
        patients with a simple csv dataset </p>

      <hr></hr>

      <p id = "instructions"> To use this tool, upload a csv file and wait for the results to be displayed </p>
      <hr></hr>

      <form id = "uploaded-file">

        <input type="file" accept=".csv" id = "FileUpload" required />
        <br></br>
        <input type = "submit" value = "Run Diagnostic" id = "SubmitButton" />    
      </form>

    </div>
    
  )
  
}

const uploadForm = document.querySelector("#uploaded-file")

uploadForm.addEventListener("submit", function(e) {
  e.preventDefault ()
  let file = e.target.uploaded-file.files[0]


})

export default App