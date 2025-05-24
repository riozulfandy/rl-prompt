# Create the directory if it doesn't exist
$destination = "examples/few-shot-classification/data/16-shot"
if (!(Test-Path -Path $destination)) {
    New-Item -ItemType Directory -Path $destination -Force
}

# Download the dataset zip file from Kaggle
curl -L -o "$destination/dataset.zip" "https://www.kaggle.com/api/v1/datasets/download/riozulfandy04/rl-prompt-16-shot-classification-dataset"

# Extract the zip file
Expand-Archive -Force "$destination/dataset.zip" -DestinationPath $destination

# Delete the zip file after extraction
Remove-Item "$destination/dataset.zip"

Write-Host "Dataset downloaded, extracted, and zip file removed successfully."