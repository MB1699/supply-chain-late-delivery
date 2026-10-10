-- Databricks notebook source
-- 00 - Project setup: catalog, schemas, and volume

CREATE CATALOG IF NOT EXISTS supply_chain;

-- COMMAND ----------

CREATE SCHEMA IF NOT EXISTS supply_chain.landing;
CREATE SCHEMA IF NOT EXISTS supply_chain.bronze;
CREATE SCHEMA IF NOT EXISTS supply_chain.silver;
CREATE SCHEMA IF NOT EXISTS supply_chain.gold;

-- COMMAND ----------

CREATE VOLUME IF NOT EXISTS supply_chain.landing.raw_files;