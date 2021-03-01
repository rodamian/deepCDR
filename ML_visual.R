library(shiny)
library(tidyverse)
options(shiny.maxRequestSize=30*1024^2)
library(DT)
library(reticulate)
library(pROC)
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
                        multiple = T),
            actionButton("run", "Predict specific variants - Run ML model"),
            actionButton("plot", "Plot repertoire metrics")
        ),
        mainPanel(plotOutput("AUC", width = "700px", height = "500px")),
    ),
    fluidRow(column(9, DT::dataTableOutput('seq_table'))),
            tabPanel("Plot", column(6, plotOutput("lev_dist")),
                             column(7, plotOutput("repertoire"))
             )
)

server <- function(input, output, session) {
    observe({
        file <- input$file1
        ext <- tools::file_ext(file$datapath)

        req(file)
        validate(need(ext == "csv", "Please upload a csv file"))

        features <- read.csv(file$datapath, header = input$header)

        # Render table
        output$seq_table <- DT::renderDataTable(
            features, options = list(lengthMenu = c(5, 10, 20, 50, 100, 200),
                                     pageLength = 20,
                                     columnDefs = list(list(visible=FALSE,
                                                            targets=grep("(?i)all", colnames(features)))
                                     )))

            observe({
                x <- colnames(read.csv(input$file1$datapath, header = input$header))

                # Can use character(0) to remove all choices
                if (is.null(x))
                    x <- character(0)

                # Can also set the label and select items
                updateSelectInput(session, "to_use",
                                  choices = x
                )
            })

        # Run model
        observeEvent(input$run, {
            file <- input$file1
            ext <- tools::file_ext(file$datapath)
            features <- read.csv(file$datapath, header = input$header)

            source_python("python/ML_models.py")
            prediction <- run_models(file = features,
                       model = input$model_selected,
                       enc = input$encoding,
                       to_use = input$to_use)

            ROC <- roc(as.numeric(prediction[[2]]), prediction[[1]], ci = T)

            # Confidence intervals
            # ci.sp <- lapply(rocs, ci.sp, sensitivities=seq(0, 1, .01), boot.n=100)
            # conf <- lapply(ROC, ci)

            output$AUC <- renderPlot(ggroc(ROC) +
                geom_abline(slope=1, linetype = "dashed", color = "grey", intercept = 1) +
                ggtitle(paste0("encoding: ", input$encoding)) +
                annotate(geom = "text", x = 0.25, y = 0.2, size=6,
                         label = paste("AUC: ", as.character(round(ROC$auc, 3))))
            )
        })

        observeEvent(input$plot, {
            require(ggplot2); theme_set(theme_bw())

            # Clonal expansion
            output$repertoire <- renderPlot(
                ggplot(filter(features, cloneId < 1000), aes(x=cloneId, y=cloneFraction, fill=label)) + geom_col())

            # LV distance
            library(stringdist)
            cdr3_drug <- "CSRWGGDGFYAMDYW"
            features$lv_dist <- stringdist::stringdist(features$aaSeqCDR3, cdr3_drug, c("lv"))
            output$lev_dist <- renderPlot(ggplot(features, aes(x=lv_dist, fill=label)) + geom_histogram())
        })
    })
}
shinyApp(ui, server)

