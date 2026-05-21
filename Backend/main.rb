require 'csv'

CSV.foreach('./dbs/Leukemia_GSE9476.csv', headers: true) do |row|
    puts row ['age']
    puts row ['gene data']
    puts row ['cancer risk assessment number']
end

