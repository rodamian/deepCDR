library(shiny)
library(tidyverse)
options(shiny.maxRequestSize=30*1024^2)
library(DT)
library(reticulate)
library(keras)

ui <- bootstrapPage(
    sidebarLayout(
        sidebarPanel(
            fileInput("file1", "Choose CSV File", accept = ".csv"),
            checkboxInput("header", "Header", TRUE),
            selectInput("model_selected", "Model:",
                        c("Support vector machine" = "SVM",
                          "XGBoost" = "XGB",
                          "Neural network" = "NN")),
            selectInput("encoding", "Encoding:",
                        c("kmer" = "kmer",
                          "one hot" = "onehot")),
            selectInput("to_use", "Feature to use:",
                        c("amino acid sequence CDR3" = "aaSeqCDR3"),
                        multiple = T)
        ),
        mainPanel(plotOutput("AUC", height = 50, width = 50),
                tableOutput("contents"))
    ),
    fluidRow(column(9, DT::dataTableOutput('seq_table')),
        actionButton("run", "Predict specific variants - Run ML model"),
        actionButton("plot", "Plot repertoire metrics")),
            tabPanel("Plot", column(6, plotOutput("importance")),
                             column(7, plotOutput("repertoire"))
             )
)

server <- function(input, output, session) {
    output$contents <- renderTable({
        file <- input$file1
        ext <- tools::file_ext(file$datapath)

        req(file)
        validate(need(ext == "csv", "Please upload a csv file"))

        # Read data
        outVar = reactive({
            features <- read.csv(file$datapath, header = input$header)
            possible_to_use <- colnames(features)
                })
            observe({
                updateSelectInput(session, "to_use",
                                  choices = outVar()
            )})

        # Render table
        output$seq_table <- DT::renderDataTable(
            features, options = list(lengthMenu = c(5, 10, 20, 50, 100, 200),
                                     pageLength = 20,
                                     columnDefs = list(list(visible=FALSE,
                                                            targets=grep("(?i)all", colnames(features)))
                                                            )))
        # Run model
        observeEvent(input$run, {
            file <- input$file1
            ext <- tools::file_ext(file$datapath)
            features <- read.csv(file$datapath, header = input$header)

            source_python("python/ML_models.py")
            run_models(file = features,
                       model = input$model_selected,
                       enc = input$encoding,
                       to_use = input$to_use)
            output$AUC <- renderImage(outfile <- list(src = "AUC.png",
                                           alt = "text"), deleteFile = T)
        })

        observeEvent(input$plot, {
            require(ggplot2); theme_set(theme_bw())
            require(tidyverse)

        })
    })
}
shinyApp(ui, server)

