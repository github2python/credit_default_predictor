import React, { useState } from "react";
import "../styles/credit_score.css";
import {
  Container,
  TextField,
  Button,
  Typography,
  Box,
  Grid,
} from "@mui/material";
import axios from "axios";

function CreditScoreForm() {
  const initialState = {
    RevolvingUtilizationOfUnsecuredLines: "",
    age: "",
    NumberOfTime30_59DaysPastDueNotWorse: "",
    DebtRatio: "",
    MonthlyIncome: "",
    NumberOfOpenCreditLinesAndLoans: "",
    NumberOfTimes90DaysLate: "",
    NumberRealEstateLoansOrLines: "",
    NumberOfTime60_89DaysPastDueNotWorse: "",
    NumberOfDependents: "",
  };
  const [features, setFeatures] = useState(initialState);
  const [prediction, setPrediction] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;

    setFeatures({
      ...features,
      [name]: value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post("http://localhost:5000/api/predict", {
        features,
      });
      setPrediction(response.data.prediction);
      setFeatures(initialState);
    } catch (error) {
      console.error(
        "Error getting prediction:",
        error.response ? error.response.data : error.message
      );
    }
  };

  return (
    <Container maxWidth="sm" style={{ marginTop: "50px" }}>
      <Box
        sx={{
          boxShadow: 3,
          padding: 4,
          borderRadius: 2,
          backgroundColor: "#fff",
        }}
      >
        <Typography
          variant="h4"
          gutterBottom
          align="center"
          sx={{
            color: "#1976d2",
            fontWeight: "bold",
            letterSpacing: "1.5px",
            fontSize: "2.2rem",
            marginBottom: "20px",
            textShadow: "2px 2px 4px rgba(1, 1, 1, 0.2)",
          }}
        >
          Credit Score Prediction
        </Typography>
        <form onSubmit={handleSubmit}>
          <Grid container spacing={2}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Revolving Utilization Of Unsecured Lines"
                variant="outlined"
                type="number"
                name="RevolvingUtilizationOfUnsecuredLines"
                value={features.RevolvingUtilizationOfUnsecuredLines}
                onChange={handleInputChange}
                required
                helperText="Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Age"
                variant="outlined"
                type="number"
                name="age"
                value={features.age}
                onChange={handleInputChange}
                required
                helperText="Age of borrower in years."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Debt Ratio"
                variant="outlined"
                type="number"
                name="DebtRatio"
                value={features.DebtRatio}
                onChange={handleInputChange}
                required
                helperText="Debt ratio of borrower."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Monthly Income"
                variant="outlined"
                type="number"
                name="MonthlyIncome"
                value={features.MonthlyIncome}
                onChange={handleInputChange}
                required
                helperText="Monthly income of borrower."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Of Open Credit Lines And Loans"
                variant="outlined"
                type="number"
                name="NumberOfOpenCreditLinesAndLoans"
                value={features.NumberOfOpenCreditLinesAndLoans}
                onChange={handleInputChange}
                required
                helperText="Number of open credit lines and loans."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Of Times 90 Days Late"
                variant="outlined"
                type="number"
                name="NumberOfTimes90DaysLate"
                value={features.NumberOfTimes90DaysLate}
                onChange={handleInputChange}
                required
                helperText="Number of times borrower has been 90 days or more past due."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Of Dependents"
                variant="outlined"
                type="number"
                name="NumberOfDependents"
                value={features.NumberOfDependents}
                onChange={handleInputChange}
                required
                helperText="Number of dependents in the family excluding themselves."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Real Estate Loans Or Lines"
                variant="outlined"
                type="number"
                name="NumberRealEstateLoansOrLines"
                value={features.NumberRealEstateLoansOrLines}
                onChange={handleInputChange}
                required
                helperText="Number of mortgage and real estate loans including home equity lines of credit."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Of Time 60-89 Days Past Due Not Worse"
                variant="outlined"
                type="number"
                name="NumberOfTime60_89DaysPastDueNotWorse"
                value={features.NumberOfTime60_89DaysPastDueNotWorse}
                onChange={handleInputChange}
                required
                helperText="Number of times borrower has been 60-89 days past due but no worse in the last 2 years."
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Number Of Time 30-59 Days Past Due Not Worse"
                variant="outlined"
                type="number"
                name="NumberOfTime30_59DaysPastDueNotWorse"
                value={features.NumberOfTime30_59DaysPastDueNotWorse}
                onChange={handleInputChange}
                required
                helperText="Number of times borrower has been 30-59 days past due but no worse in the last 2 years."
              />
            </Grid>
            <Grid item xs={12}>
              <Button
                fullWidth
                type="submit"
                variant="contained"
                color="primary"
              >
                Predict
              </Button>
            </Grid>
          </Grid>
        </form>
        {prediction != null && (
          <div className="result">
            {prediction === 0 && (
              <h2>
                The prediction for the person to have credit default in near
                future is <b className="neg">Negative</b>
              </h2>
            )}
            {prediction === 1 && (
              <h2>
                The prediction for the person to have credit default in near
                future is <b className=" pos">Positive</b>
              </h2>
            )}
          </div>
        )}
      </Box>
    </Container>
  );
}

export default CreditScoreForm;
