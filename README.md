This is code for the NeurIPS 2018 paper "Autoconj: Recognizing and Exploiting Conjugacy Without a Domain-Specific Language" by Matthew D Hoffman, Matthew J Johnson, and Dustin Tran.

Deriving conditional and marginal distributions using conjugacy relationships can be time consuming and error prone. In this project, we propose a strategy for automating such derivations. Unlike previous systems which focus on relationships between pairs of random variables, our system (which we call /AutoConj/) operates directly on Python functions that compute log-joint distribution functions. Autoconj provides support for conjugacy-exploiting algorithms in any Python-embedded PPL. This paves the way for accelerating development of novel inference algorithms and structure-exploiting modeling strategies.

This is not an official Google product.
