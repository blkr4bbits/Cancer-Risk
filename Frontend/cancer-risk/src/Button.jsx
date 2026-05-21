

function Button () { return (
    <form action = "/action_page.php">
        <input type = "file" id = "myFile" name = "filename" accept= ".csv"></input>
        <input type = "submit"></input>
    </form>
    )
}

export default Button