post '/predict' do
  file = params[:file]

  result = `python3 predictor.py #{file.path}`

  return result
end